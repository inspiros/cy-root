#include "native/buffers.hpp"

#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <new>
#include <stdexcept>
#include <type_traits>
#include <utility>

// Track only array allocations made by the owner's new[]/delete[] pair.
// This verifies cleanup even on hosts where LeakSanitizer cannot attach.
std::size_t live_array_allocations = 0;
bool fail_next_array_allocation = false;

void* operator new[](std::size_t bytes) {
    if (std::exchange(fail_next_array_allocation, false)) {
        throw std::bad_alloc();
    }
    if (void* allocation = std::malloc(bytes)) {
        ++live_array_allocations;
        return allocation;
    }
    throw std::bad_alloc();
}

void operator delete[](void* allocation) noexcept {
    if (allocation != nullptr) {
        --live_array_allocations;
        std::free(allocation);
    }
}

void operator delete[](void* allocation, std::size_t) noexcept {
    ::operator delete[](allocation);
}

using cyroot::native::MatrixView;
using cyroot::native::OwnedDoubleBuffer;
using cyroot::native::VectorView;

static_assert(!std::is_copy_constructible<OwnedDoubleBuffer>::value, "owner must not copy");
static_assert(!std::is_copy_assignable<OwnedDoubleBuffer>::value, "owner must not copy");
static_assert(std::is_nothrow_move_constructible<OwnedDoubleBuffer>::value, "move must be noexcept");
static_assert(std::is_nothrow_move_assignable<OwnedDoubleBuffer>::value, "move must be noexcept");
static_assert(std::is_nothrow_destructible<OwnedDoubleBuffer>::value, "destruction must be noexcept");
static_assert(std::is_copy_constructible<VectorView>::value, "view metadata should copy");
static_assert(std::is_copy_constructible<MatrixView>::value, "view metadata should copy");

// Abort even if a release build is accidentally compiled with NDEBUG.
#undef assert
#define assert(expression) do { if (!(expression)) std::abort(); } while (false)

template<class Exception, class Function>
void must_throw(Function function) {
    bool caught = false;
    try {
        function();
    } catch (const Exception&) {
        caught = true;
    }
    assert(caught);
}

void test_ownership() {
    OwnedDoubleBuffer empty;
    assert(empty.empty() && empty.data() == nullptr && empty.size_bytes() == 0);
    assert(empty.view().empty());
    const auto baseline_allocations = live_array_allocations;
    for (int repeat = 0; repeat < 1000; ++repeat) {
        OwnedDoubleBuffer first(6);
        assert(live_array_allocations == baseline_allocations + 1);
        assert(first.size() == 6 && first.size_bytes() == 6 * sizeof(double));
        for (std::size_t index = 0; index < first.size(); ++index) {
            assert(first.get(index) == 0.0);
            first.set(index, static_cast<double>(index + 1));
        }
        const double* allocation = first.data();
        OwnedDoubleBuffer second(std::move(first));
        assert(live_array_allocations == baseline_allocations + 1);
        assert(first.empty() && first.data() == nullptr);
        assert(second.data() == allocation && second.get(5) == 6.0);
        OwnedDoubleBuffer third(20);
        assert(live_array_allocations == baseline_allocations + 2);
        third = std::move(second);  // Frees third's previous allocation.
        assert(live_array_allocations == baseline_allocations + 1);
        assert(second.empty() && second.data() == nullptr);
        assert(third.data() == allocation && third.get(5) == 6.0);
        auto* self_alias = &third;
        third = std::move(*self_alias);  // Self-move preserves ownership.
        assert(third.data() == allocation);
        const auto matrix = third.matrix_view(2, 3);
        assert(matrix.get(1, 2) == 6.0);
        third.fill(-2.5);
        assert(matrix.get(0, 1) == -2.5);  // View borrows the existing allocation.
        must_throw<std::invalid_argument>([&] { third.matrix_view(3, 3); });
        must_throw<std::out_of_range>([&] { third.get(6); });
        must_throw<std::out_of_range>([&] { third.set(6, 2.0); });
        third.reset();
        assert(third.empty() && third.data() == nullptr);
        assert(live_array_allocations == baseline_allocations);
        { OwnedDoubleBuffer destroyed_at_scope_exit(5); }
        assert(live_array_allocations == baseline_allocations);
    }
    must_throw<std::length_error>([] { OwnedDoubleBuffer(OwnedDoubleBuffer::max_size() + 1); });
    assert(live_array_allocations == baseline_allocations);
    fail_next_array_allocation = true;
    must_throw<std::bad_alloc>([] { OwnedDoubleBuffer(4); });
    assert(live_array_allocations == baseline_allocations);
    {
        OwnedDoubleBuffer original(3);
        original.fill(7.0);
        const auto* original_pointer = original.data();
        fail_next_array_allocation = true;
        must_throw<std::bad_alloc>([&] { original = OwnedDoubleBuffer(9); });
        assert(original.data() == original_pointer && original.get(2) == 7.0);
        assert(live_array_allocations == baseline_allocations + 1);
    }
    assert(live_array_allocations == baseline_allocations);
}

void test_strides_and_metadata() {
    const std::array<double, 6> storage{{1., 2., 3., 4., 5., 6.}};
    VectorView forward(storage.data(), 3, 2 * sizeof(double));
    VectorView reverse(storage.data() + 5, 6, -static_cast<std::ptrdiff_t>(sizeof(double)));
    VectorView broadcast(storage.data() + 2, 100, 0);
    assert(forward.get(2) == 5.0 && reverse.get(5) == 1.0 && broadcast.get(99) == 3.0);
    auto metadata_copy = reverse;
    reverse = VectorView();
    assert(metadata_copy.size() == 6 && metadata_copy.stride_bytes() == -8);
    assert(metadata_copy.get(0) == 6.0);
    MatrixView transposed(storage.data(), 3, 2, sizeof(double), 3 * sizeof(double));
    assert(transposed.get(2, 1) == 6.0 && transposed.get(1, 0) == 2.0);
    MatrixView flipped(storage.data() + 5, 2, 3, -3 * static_cast<std::ptrdiff_t>(sizeof(double)),
                       -static_cast<std::ptrdiff_t>(sizeof(double)));
    assert(flipped.get(1, 2) == 1.0);
    MatrixView repeated(storage.data(), 3, 4, 0, 0);
    assert(repeated.get(2, 3) == 1.0);
    must_throw<std::out_of_range>([&] { forward.get(3); });
    must_throw<std::out_of_range>([&] { transposed.get(3, 0); });
    must_throw<std::out_of_range>([&] { transposed.get(0, 2); });

    const auto bounded = VectorView::from_span(storage.data(), sizeof(storage), 5 * sizeof(double),
                                               6, -static_cast<std::ptrdiff_t>(sizeof(double)));
    assert(bounded.get(5) == 1.0);
    const auto bounded_matrix = MatrixView::from_span(storage.data(), sizeof(storage), 0,
                                                      2, 3, 3 * sizeof(double), sizeof(double));
    assert(bounded_matrix.get(1, 2) == 6.0);
    must_throw<std::out_of_range>([&] {
        VectorView::from_span(storage.data(), sizeof(storage), 0, 2, -8);
    });
    must_throw<std::out_of_range>([&] {
        MatrixView::from_span(storage.data(), sizeof(storage), 0, 2, 4, 24, 8);
    });
    must_throw<std::invalid_argument>([&] {
        VectorView::from_span(storage.data(), sizeof(storage), sizeof(storage) + 1, 0, 0);
    });
}

void test_unaligned_and_empty() {
    std::array<unsigned char, 19> bytes{};
    const double first = 1.25, second = -3.75;
    std::memcpy(bytes.data() + 1, &first, sizeof(double));
    std::memcpy(bytes.data() + 10, &second, sizeof(double));
    VectorView unaligned(bytes.data() + 1, 2, 9);
    assert(unaligned.get(0) == first && unaligned.get(1) == second);
    VectorView empty(nullptr, 0, std::numeric_limits<std::ptrdiff_t>::min());
    MatrixView no_columns(nullptr, std::numeric_limits<std::size_t>::max(), 0,
                          std::numeric_limits<std::ptrdiff_t>::min(), 8);
    MatrixView no_rows(nullptr, 0, 10, 8, std::numeric_limits<std::ptrdiff_t>::max());
    assert(empty.empty() && no_columns.empty() && no_rows.empty());
    assert(VectorView::from_span(nullptr, 0, 0, 0, 8).empty());
    assert(MatrixView::from_span(nullptr, 0, 0, 0, 2, 16, 8).empty());
    must_throw<std::out_of_range>([&] { empty.get(0); });
    must_throw<std::out_of_range>([&] { no_columns.get(0, 0); });
    must_throw<std::invalid_argument>([] { VectorView(nullptr, 1, 8); });
    must_throw<std::invalid_argument>([] { MatrixView(nullptr, 1, 1, 8, 8); });
}

void test_offset_failures() {
    double value = 1.0;
    constexpr auto largest = std::numeric_limits<std::ptrdiff_t>::max();
    constexpr auto smallest = std::numeric_limits<std::ptrdiff_t>::min();
    must_throw<std::length_error>([&] {
        VectorView(&value, std::numeric_limits<std::size_t>::max(), 0);
    });
    must_throw<std::length_error>([&] {
        MatrixView(&value, std::numeric_limits<std::size_t>::max(), 2, 0, 0);
    });
    must_throw<std::overflow_error>([&] { VectorView(&value, 3, largest); });
    must_throw<std::overflow_error>([&] { VectorView(&value, 2, smallest); });
    must_throw<std::overflow_error>([&] { MatrixView(&value, 2, 2, largest, largest); });
    must_throw<std::overflow_error>([&] { MatrixView(&value, 2, 2, smallest, smallest); });
    must_throw<std::overflow_error>([] {
        const auto address = std::numeric_limits<std::uintptr_t>::max() - 3;
        VectorView(reinterpret_cast<const void*>(address), 1, 8);
    });
    must_throw<std::overflow_error>([] {
        VectorView(reinterpret_cast<const void*>(1), 2, -8);
    });
    // A stride is irrelevant on a singleton axis and must not be negated/multiplied.
    assert(VectorView(&value, 1, smallest).get(0) == value);
    assert(MatrixView(&value, 1, 1, smallest, largest).get(0, 0) == value);
}

int main() {
    test_ownership();
    test_strides_and_metadata();
    test_unaligned_and_empty();
    test_offset_failures();
}
