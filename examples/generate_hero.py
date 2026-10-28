"""Generate the README's vector hero from recorded CyRoot solver histories.

Run: python examples/generate_hero.py
Requires the examples extra. The SVG embeds its font outlines and gradients;
its geometry comes from Newton and Powell dogleg on the gallery problems.
"""
import argparse
from io import StringIO
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

if __package__:
    from .scalar_steps import function, derivative, record_histories as scalar_histories
    from .vector_steps import residual, record_histories as vector_histories
else:
    from scalar_steps import function, derivative, record_histories as scalar_histories
    from vector_steps import residual, record_histories as vector_histories

SVG = 'http://www.w3.org/2000/svg'
XLINK = 'http://www.w3.org/1999/xlink'
CREAM = '#f3f0df'
GOLD = '#ddc08a'
MINT = '#a9cda2'


def create_hero():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    scalar = scalar_histories(('newton',))['Newton']
    vector = next(iter(vector_histories(('powell_dogleg',)).values()))
    if not scalar[-1].residual_converged or not vector[-1].residual_converged:
        raise RuntimeError('Hero histories must reach their residual roots.')
    with matplotlib.rc_context({'svg.fonttype': 'path', 'svg.hashsalt': 'cyroot-hero',
                                'font.family': 'DejaVu Sans', 'mathtext.fontset': 'stix'}):
        fig = plt.figure(figsize=(14., 4.7), dpi=100, facecolor='none')
        # A subtle system of residual contours and its actual dogleg path.
        field = fig.add_axes((.33, .36, .74, .90), facecolor='none')
        xx, yy = np.meshgrid(np.linspace(-1.6, 1.55, 240), np.linspace(-.5, 1.9, 210))
        values = residual((xx, yy))
        norm = np.maximum(np.abs(values[0]), np.abs(values[1]))
        field.contour(xx, yy, norm, levels=(.125, .25, .5, 1., 2., 4., 7., 11., 16., 24.),
                      colors=MINT, linewidths=.8, alpha=.21)
        points = np.array([state.root for state in vector])
        field.plot(points[:, 0], points[:, 1], color=MINT, lw=1.8, alpha=.68)
        field.scatter(points[::2, 0], points[::2, 1], s=9., color=MINT, alpha=.75)
        field.scatter(*points[-1], s=44., facecolor=MINT, edgecolor='none', zorder=5)
        field.text(points[-1, 0]-.7, points[-1, 1]+.02, r'$F(\mathbf{x}_\star)=\mathbf{0}$',
                   color=MINT, fontsize=12, alpha=.75)
        field.set(xlim=(-1.6, 1.55), ylim=(-.5, 1.9));field.axis('off')

        # Scalar zero line, cubic, and the first three recorded Newton tangents.
        axis = fig.add_axes((.35, -.17, .74, 1.39), facecolor='none')
        axis.set(xlim=(1.15, 2.), ylim=(-2.3, 3.6))
        for x in (1.3, 1.5, 1.7, 1.9):axis.axvline(x, color=CREAM, lw=.5, alpha=.095)
        for y in (-1., 1., 2., 3., 4.):axis.axhline(y, color=CREAM, lw=.5, alpha=.095)
        axis.axhline(0., color=GOLD, lw=1.5, alpha=.72)
        xs=np.linspace(1.10, 2.10, 400)
        axis.plot(xs, function(xs), color=CREAM, lw=3., alpha=.96)
        for i,state in enumerate(scalar[:3]):
            x,y=state.root,state.f_root
            intercept=x-y/derivative(x)
            alpha=(.8,.65,.48)[i]
            axis.plot((x,intercept),(y,0.), color=GOLD, lw=1.9, alpha=alpha)
            axis.plot((intercept,intercept),(0.,function(intercept)),
                      color=GOLD, lw=.85, alpha=alpha*.7, dashes=(2.,3.))
            axis.scatter(x,y,s=25.-4.*i,color=CREAM,edgecolor='none',alpha=.9,zorder=5)
        root=scalar[-1].root
        axis.scatter(root,0.,s=650.,color=GOLD,alpha=.09,edgecolor='none',zorder=6)
        axis.scatter(root,0.,s=120.,facecolor='#173b2e',edgecolor=GOLD,lw=1.5,zorder=7)
        axis.scatter(root,0.,s=19.,color=GOLD,edgecolor='none',zorder=8)
        axis.text(root+.025,-.52,r'$f(x_\star)=0$',color=GOLD,fontsize=19)
        axis.axis('off')

        fig.text(.054,.79,'Root Finding In Cython',color=MINT,fontsize=10.5)
        fig.text(.051,.59,'CyRoot',fontfamily='DejaVu Serif',fontsize=65,color=CREAM)
        fig.text(.054,.425,'Where Functions Meet Zero',fontsize=21.5,color=CREAM)
        fig.text(.055,.31,'Python Functions. Native Callbacks.',fontsize=11.7,color='#c2d2b9')
        fig.text(.055,.135,'Scalar Equations  ·  Nonlinear Systems',fontsize=11,color=MINT)
        stream=StringIO()
        fig.savefig(stream,format='svg',transparent=True,metadata={'Date':None})
        plt.close(fig)

    # Native SVG gradients keep the asset scalable and self-contained.
    ET.register_namespace('',SVG);ET.register_namespace('xlink',XLINK)
    root=ET.fromstring(stream.getvalue())
    _,_,width,height=map(float,root.attrib['viewBox'].split())
    root.set('width','1400');root.set('height','470')
    title=ET.Element(f'{{{SVG}}}title');title.text='CyRoot — Where Functions Meet Zero'
    desc=ET.Element(f'{{{SVG}}}desc')
    desc.text=('Actual Newton tangents approach the zero of x cubed minus x minus two. '
               'A recorded Powell dogleg path crosses residual contours for a nonlinear system. '
               'This illustration shows convergence geometry, not a timing comparison.')
    root.insert(0,title);root.insert(1,desc)
    defs=root.find(f'{{{SVG}}}defs')
    gradient=ET.SubElement(defs,f'{{{SVG}}}linearGradient',id='forest',x1='0',y1='0',x2='1',y2='.8')
    for offset,color in (('0%','#102c2c'),('43%','#184835'),('76%','#284e36'),('100%','#314432')):
        ET.SubElement(gradient,f'{{{SVG}}}stop',offset=offset,attrib={'stop-color':color})
    washes=(('teal','.14','.02','.67','#4c9290','.26'),
            ('jade','.58','-.12','.72','#a5c496','.16'),
            ('gold','.88','.87','.73','#c5a35e','.28'),
            ('root-glow','.68','.64','.38','#d7c284','.12'))
    for name,cx,cy,r,color,opacity in washes:
        glow=ET.SubElement(defs,f'{{{SVG}}}radialGradient',id=name,cx=cx,cy=cy,r=r)
        for offset,alpha in (('0%',opacity),('100%','0')):
            ET.SubElement(glow,f'{{{SVG}}}stop',offset=offset,
                          attrib={'stop-color':color,'stop-opacity':alpha})
    grain=ET.SubElement(defs,f'{{{SVG}}}filter',id='grain',x='0',y='0',width='100%',height='100%')
    ET.SubElement(grain,f'{{{SVG}}}feTurbulence',type='fractalNoise',baseFrequency='.7',
                  numOctaves='3',seed='19',stitchTiles='stitch')
    ET.SubElement(grain,f'{{{SVG}}}feColorMatrix',type='saturate',values='0')
    index=list(root).index(root.find(f'{{{SVG}}}g'))
    for fill in ['url(#forest)']+[f'url(#{wash[0]})' for wash in washes]:
        root.insert(index,ET.Element(f'{{{SVG}}}rect',width=str(width),height=str(height),fill=fill))
        index+=1
    root.insert(index,ET.Element(f'{{{SVG}}}rect',width=str(width),height=str(height),
        fill='#ffffff',filter='url(#grain)',opacity='.035'))
    return ET.tostring(root,encoding='unicode',xml_declaration=True)+'\n'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--save',type=Path,default=Path(__file__).resolve().parents[1]/'assets/hero.svg')
    args=parser.parse_args()
    args.save.parent.mkdir(parents=True,exist_ok=True)
    args.save.write_text(create_hero(),encoding='utf-8')
    print('Saved',args.save.resolve())


if __name__=='__main__':main()
