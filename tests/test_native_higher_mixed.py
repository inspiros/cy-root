"""Runtime tensor differences: exact polynomial moments and a native ABI probe."""
import importlib
import itertools
import math
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from fractions import Fraction
from unittest.mock import patch

from cyroot.native import (HessianPlan, NativeCallbackError, RectangularCallback,
                           TensorDifferencePlan, VectorCallback)


def compositions(n, order):
    return (a for a in itertools.product(range(order+1),repeat=n) if sum(a) == order)


def slots(order, scheme):
    if not order:
        return (0,)
    if scheme != 'central':
        return tuple((1 if scheme == 'forward' else -1)*j for j in range(order+1))
    return (0,)+tuple(sign*j for j in range(1,(order+1)//2+1) for sign in (1,-1))


def graph_counts(n, order, scheme):
    points={(0,)*n}
    terms=0
    for powers in compositions(n,order):
        products=list(itertools.product(*(slots(r,scheme) for r in powers)))
        terms += len(products)
        points.update(products)
    return points,terms


def homogeneous(point,m,order):
    powers=list(compositions(len(point),order))
    return tuple(13.+o+sum((o+1)*(1+sum((j+1)*r for j,r in enumerate(a)))*
                          math.prod(x**r for x,r in zip(point,a)) for a in powers)
                 for o in range(m))


def analytic_entry(indices,output,n):
    counts=tuple(indices.count(j) for j in range(n))
    return ((output+1)*(1+sum((j+1)*r for j,r in enumerate(counts)))*
            math.prod(math.factorial(r) for r in counts))


def entries(tensor,n,order):
    for output,plane in enumerate(tensor):
        for indices in itertools.product(range(n),repeat=order):
            value=plane
            for index in indices:
                value=value[index]
            yield output,indices,value


def all_nan(tensor):
    return all(all_nan(value) if isinstance(value,tuple) else math.isnan(value)
               for value in tensor)


def moment_weights(nodes,derivative):
    """Exact moment-system elimination, independent of the Fornberg recurrence."""
    count=len(nodes)
    rows=[[Fraction(node)**p for node in nodes]+
          [Fraction(math.factorial(p) if p == derivative else 0)] for p in range(count)]
    for column in range(count):
        pivot=next(row for row in range(column,count) if rows[row][column])
        rows[column],rows[pivot]=rows[pivot],rows[column]
        factor=rows[column][column]
        rows[column]=[value/factor for value in rows[column]]
        for row in range(count):
            if row != column:
                multiplier=rows[row][column]
                rows[row]=[a-multiplier*b for a,b in zip(rows[row],rows[column])]
    return tuple(row[-1] for row in rows)


def exact_stencil(function,point,steps,powers,scheme):
    coordinates=[]
    weights=[]
    for x,h,r in zip(point,steps,powers):
        axis=tuple(x+j*h for j in slots(r,scheme))
        coordinates.append(axis)
        weights.append(moment_weights(tuple(Fraction(y)-Fraction(x) for y in axis),r))
    total=Fraction(0)
    base=Fraction(function(point)[0])
    for choices in itertools.product(*(range(len(axis)) for axis in coordinates)):
        sample=tuple(axis[j] for axis,j in zip(coordinates,choices))
        total += math.prod(row[j] for row,j in zip(weights,choices))*(Fraction(function(sample)[0])-base)
    return float(total)


class HigherMixedDifferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workspace=tempfile.TemporaryDirectory(prefix='cyroot-higher-mixed-consumer-', ignore_cleanup_errors=(os.name == 'nt'))
        cls.addClassCleanup(cls.workspace.cleanup)
        directory=Path(cls.workspace.name)
        name='native_higher_mixed_consumer'
        (directory/(name+'.pyx')).write_bytes(Path(__file__).with_name(name+'.pyx').read_bytes())
        (directory/'setup.py').write_text(
            'from setuptools import Extension,setup\n'
            'import sys\n'
            'from Cython.Build import cythonize\n'
            'from cyroot.native import get_include,get_cython_include\n'
            'flags=["/std:c++17"] if sys.platform == "win32" else ["-std=c++17"]\n'
            'setup(name="cyroot-higher-mixed-consumer",ext_modules=cythonize(\n'
            ' [Extension("native_higher_mixed_consumer",["native_higher_mixed_consumer.pyx"],\n'
            ' language="c++",include_dirs=[get_include()],extra_compile_args=flags)],\n'
            ' language_level=3,include_path=[get_cython_include()]))\n')
        environment=os.environ.copy()
        environment.pop('PYTHONPATH',None)
        build=subprocess.run([sys.executable,'setup.py','build_ext','--inplace'],cwd=directory,
                             env=environment,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        if build.returncode:
            raise AssertionError('Tensor external consumer failed:\n'+build.stdout)
        sys.path.insert(0,str(directory))
        cls.addClassCleanup(sys.path.remove,str(directory))
        cls.consumer=importlib.import_module(name)
        cls.addClassCleanup(sys.modules.pop,name,None)

    def test_polynomial_mixed_orders_permutations_rectangles_and_unique_samples(self):
        for n,m,order in ((2,1,3),(2,3,3),(3,2,3),(2,2,4),(1,3,4)):
            point=tuple(.25*(j+1) for j in range(n))
            steps=tuple((-1)**j*.25 for j in range(n))
            function=lambda x:homogeneous(x,m,order)
            for scheme in ('central','forward','backward'):
                plan=TensorDifferencePlan(n,m,order,scheme,steps)
                offsets,_=graph_counts(n,order,scheme)
                expected_points={tuple(x+offset*h for x,h,offset in zip(point,steps,row))
                                 for row in offsets}
                self.assertEqual(plan.sample_count,len(offsets)-1)
                for cached in (False,True):
                    seen=[]
                    result=plan.evaluate(lambda x:seen.append(x) or function(x),point,
                                         f_x=function(point) if cached else None)
                    with self.subTest(n=n,m=m,order=order,scheme=scheme,cached=cached):
                        self.assertEqual((result.status,result.callback_code),('ok',0))
                        self.assertEqual((result.f_calls,len(seen)),(len(offsets)-cached,)*2)
                        self.assertEqual(len(seen),len(set(seen)))
                        self.assertEqual(set(seen),expected_points-({point} if cached else set()))
                        if not cached:
                            self.assertEqual(seen[0],point)
                        by_counts={}
                        for o,indices,value in entries(result.derivative,n,order):
                            expected=analytic_entry(indices,o,n)
                            self.assertTrue(math.isclose(value,expected,rel_tol=1e-10,abs_tol=1e-8))
                            key=o,tuple(indices.count(j) for j in range(n))
                            encoded=struct.pack('d',value)
                            self.assertEqual(by_counts.setdefault(key,encoded),encoded)
                if n == m:
                    self.assertEqual(plan.evaluate(VectorCallback(function,n),point).derivative,
                                     plan.evaluate(RectangularCallback(function,n,m),point).derivative)

    def test_actual_rounded_coordinate_fraction_oracle_and_stencil_preflight(self):
        point=(1.,.5)
        tiny=3*2**-53
        unit=2**-52
        function=lambda x:(1.+(1+(x[0]-1.)/unit)**5*
                           (1+x[1]+x[1]**2+x[1]**3),)
        self.assertNotEqual(point[0]+tiny-point[0],-(point[0]-tiny-point[0]))
        for scheme in ('central','forward','backward'):
            for sign in (1,-1):
                steps=(sign*tiny,-.25)
                result=TensorDifferencePlan(2,1,3,scheme,steps).evaluate(function,point)
                self.assertEqual(result.status,'ok')
                for _,indices,value in entries(result.derivative,2,3):
                    powers=tuple(indices.count(j) for j in range(2))
                    expected=exact_stencil(function,point,steps,powers,scheme)
                    with self.subTest(scheme=scheme,sign=sign,indices=indices):
                        self.assertTrue(math.isclose(value,expected,rel_tol=2e-11,abs_tol=1e-8),
                                        (value,expected))
        for point,step,status in (((1.,1.),(1e-320,.25),'breakdown'),
                                  ((0.,0.),(1e308,.25),'nonfinite')):
            seen=[]
            result=TensorDifferencePlan(2,1,3,step=step).evaluate(
                lambda x:seen.append(x) or (1.,),point,f_x=(1.,))
            self.assertEqual((result.status,result.f_calls),(status,0))
            self.assertEqual(seen,[])
            self.assertTrue(all_nan(result.derivative))

    def test_native_plan_raw_descriptor_nogil_capacity_failure_and_restore(self):
        guards=(1234567.,1234567.,9876543.,9876543.)
        expected=tuple((o+1)*((6.,4.,6.,30.)[sum(indices)])
                       for indices in itertools.product(range(2),repeat=3) for o in range(3))
        for scheme in ('central','forward','backward'):
            plan=TensorDifferencePlan(2,3,3,scheme,(.25,-.25))
            allocation=self.consumer.allocation(plan)
            for raw in (False,True):
                result=self.consumer.compute_direct(plan,use_config=raw,repeats=2)
                self.assertEqual((result['status'],result['calls'],result['callback_code']),
                                 (0,2*plan.sample_count,0))
                for actual,wanted in zip(result['flat'],expected):
                    self.assertTrue(math.isclose(actual,wanted,rel_tol=1e-11,abs_tol=1e-9))
                self.assertEqual((result['guards'],result['restored_point']),(guards,(.5,1.)))
                self.assertEqual(result['gil_held'],0)
                self.assertTrue(result['stable'])
            self.assertEqual(self.consumer.allocation(plan),allocation)
        plan=TensorDifferencePlan(2,3,3,step=.25)
        invalid=[{'capacity':plan.workspace_size-1},{'output_capacity':23}]
        invalid += [dict(use_config=True,config_overrides=change) for change in
                    ({'n':0},{'m':0},{'order':2},{'nodes':4},{'tensor_size':23},
                     {'input_size':7},{'components':3},{'samples':0},{'terms':0},
                     {'work':plan.workspace_size-1},{'scheme':-1})]
        for options in invalid:
            with self.subTest(options=options):
                result=self.consumer.compute_direct(plan,initial_calls=7,**options)
                self.assertEqual((result['status'],result['calls'],result['callback_code']),(2,7,0))
                self.assertTrue(result['untouched_output'])
                self.assertTrue(result['untouched_workspace'])
                self.assertEqual(result['guards'],guards)
        for options,status,calls in (({'fail_after':0},3,1),({'fail_after':3},3,4),
                ({'partial_after':3},1,4),({'base':(0.,math.nan,0.)},1,0),
                ({'point':(math.inf,1.)},1,0)):
            result=self.consumer.compute_direct(plan,**options)
            self.assertEqual((result['status'],result['calls']),(status,calls))
            self.assertEqual(result['callback_code'],-19 if status == 3 else 0)
            self.assertTrue(all(math.isnan(v) for v in result['flat']))
            self.assertEqual(result['restored_point'],options.get('point',(.5,1.)))
            self.assertEqual(result['guards'],guards)
        for after,stage in ((0,'function'),(1,'derivative'),(4,'derivative')):
            owner=self.consumer.ContextOwner(fail_after=after)
            with self.assertRaises(NativeCallbackError) as caught:
                plan.evaluate(owner.callback(),(.5,1.))
            failure=caught.exception
            self.assertEqual((failure.native_status,failure.callback_stage),(-43,stage))
            self.assertEqual((failure.result.status,failure.result.f_calls),('callback_failure',after+1))
            self.assertTrue(all_nan(failure.result.derivative))
            self.assertEqual(owner.gil_held,0)

    def test_budget_properties_validation_reservations_and_exception_identity(self):
        steps=[.25,-.5]
        plan=TensorDifferencePlan(2,3,3,step=steps)
        steps[:]=[7.,8.]
        points,terms=graph_counts(2,3,'central')
        q=math.comb(4,3)
        nodes=5
        work=len(points)*3+4+q*3+3*nodes+3+nodes*4+2*nodes
        metadata=(q*3+q+1+terms+8)*struct.calcsize('P')+len(points)*2*struct.calcsize('i')
        self.assertEqual((plan.input_dimension,plan.output_dimension,plan.order,plan.shape),
                         (2,3,3,(3,2,2,2)))
        self.assertEqual((plan.scheme,plan.step,plan.sample_count),('central',(.25,-.5),len(points)-1))
        self.assertEqual((plan.storage_bytes,plan.workspace_size,plan.output_bytes),
                         (metadata+16,work,24*8))
        self.assertEqual(plan.workspace_bytes,work*8)
        self.assertEqual(plan.evaluation_bytes,(2+3+24+work)*8)
        self.assertEqual(plan.memory_bytes,plan.storage_bytes+plan.evaluation_bytes)
        for name in ('input_dimension','output_dimension','order','shape','scheme','step',
                     'sample_count','storage_bytes','workspace_size','workspace_bytes',
                     'output_bytes','evaluation_bytes','memory_bytes'):
            with self.assertRaises((AttributeError,TypeError)):
                setattr(plan,name,None)
        for args,options in (((0,3,3),{}),((2,0,3),{}),((2,3,2),{}),((2,3,True),{}),
                ((2,3,3),{'step':0.}),((2,3,3),{'step':(.1,math.inf)}),
                ((2,3,3),{'max_nodes':4}),((2,3,3),{'max_terms':terms-1}),
                ((2,3,3),{'memory_bytes':64}),((2,3,3),{'scheme':'unknown'}),
                ((10**9,3,64),{}),((2,3,10**9),{})):
            with self.subTest(args=args,options=options),self.assertRaises((ValueError,TypeError,OverflowError)):
                TensorDifferencePlan(*args,**options)
        exact=TensorDifferencePlan(2,3,3,step=plan.step,memory_bytes=plan.memory_bytes)
        self.assertEqual(exact.memory_bytes,plan.memory_bytes)
        with self.assertRaises(ValueError):
            TensorDifferencePlan(2,3,3,step=plan.step,memory_bytes=plan.memory_bytes-1)
        # A failed explicit initialization may leave allocated graph vectors.
        # Retrying with another scheme must clear old zero-offset positions.
        retry=TensorDifferencePlan.__new__(TensorDifferencePlan)
        with self.assertRaises(ValueError):retry.__init__(2,3,3,step=0.)
        retry.__init__(2,3,3,'forward',step=.25)
        reference=TensorDifferencePlan(2,3,3,'forward',step=.25)
        function=lambda x:homogeneous(x,3,3)
        self.assertEqual(retry.evaluate(function,(.5,1.)),reference.evaluate(function,(.5,1.)))
        adapter=RectangularCallback(function,2,3)
        case=self
        attempted=[]
        class Point:
            def __len__(self):
                with case.assertRaises(RuntimeError):
                    plan.evaluate(adapter,(0.,0.))
                attempted.append(True)
                return 2
            def __getitem__(self,index):
                return (.5,1.)[index]
        self.assertEqual(plan.evaluate(adapter,Point()).status,'ok')
        self.assertTrue(attempted)
        module=importlib.import_module('cyroot.native.higher_mixed')
        original=module.TensorDerivativeEstimate
        def estimate(*args):
            with self.assertRaises(RuntimeError):
                plan.evaluate(adapter,(0.,0.))
            return original(*args)
        with patch.object(module,'TensorDerivativeEstimate',estimate):
            self.assertEqual(plan.evaluate(adapter,(.5,1.)).status,'ok')
        failure=KeyboardInterrupt('mixed sample')
        def exploding(x):
            if x != (.5,1.):
                raise failure
            return function(x)
        with self.assertRaises(KeyboardInterrupt) as caught:
            plan.evaluate(exploding,(.5,1.))
        self.assertIs(caught.exception,failure)
        self.assertEqual(plan.evaluate(adapter,(.5,1.)).status,'ok')
        for invalid in (HessianPlan(2),RectangularCallback(function,2,3,jacobian=True),
                        RectangularCallback(function,3,2),VectorCallback(function,2)):
            with self.assertRaises((TypeError,ValueError)):
                plan.evaluate(invalid,(.5,1.))
        uninitialized=TensorDifferencePlan.__new__(TensorDifferencePlan)
        with self.assertRaises(RuntimeError):
            uninitialized.evaluate(function,(.5,1.))
        seen=[]
        result=plan.evaluate(lambda x:seen.append(x) or function(x),(math.inf,1.))
        self.assertEqual((result.status,result.f_calls,seen),('nonfinite',0,[]))
        self.assertTrue(all_nan(result.derivative))

    def test_runtime_orders_five_six_and_raised_node_cap_without_symbolic_tables(self):
        for order in (5,6):
            for scheme in ('central','forward','backward'):
                plan=TensorDifferencePlan(2,1,order,scheme,step=.5)
                result=plan.evaluate(lambda x:homogeneous(x,1,order),(0.,0.))
                self.assertEqual(result.status,'ok')
                for o,indices,value in entries(result.derivative,2,order):
                    self.assertTrue(math.isclose(value,analytic_entry(indices,o,2),
                                                 rel_tol=1e-8,abs_tol=2e-5))
        with self.assertRaises(ValueError):
            TensorDifferencePlan(1,1,66)
        raised=TensorDifferencePlan(1,1,66,max_nodes=67)
        self.assertEqual((raised.order,raised.shape,raised.sample_count),(66,(1,)*67,66))
        # A raised resource limit permits planning; this deliberately makes no
        # accuracy claim for a numerically ill-conditioned derivative of order66.
        constant=raised.evaluate(lambda x:(1.,),(0.,))
        self.assertEqual((constant.status,constant.derivative),('ok',nest_single_zero(67)))


def nest_single_zero(depth):
    value=0.
    for _ in range(depth):
        value=(value,)
    return value


if __name__ == '__main__':
    unittest.main()
