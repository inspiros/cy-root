#ifndef CYROOT_NATIVE_BUFFERS_HPP
#define CYROOT_NATIVE_BUFFERS_HPP

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>

namespace cyroot {
namespace native {

namespace detail {

inline std::size_t checked_count(std::size_t count) {
    constexpr auto maximum = static_cast<std::size_t>(
        std::numeric_limits<std::ptrdiff_t>::max()) / sizeof(double);
    if (count > maximum) {
        throw std::length_error("double buffer byte count exceeds PTRDIFF_MAX");
    }
    return count;
}

inline std::size_t checked_product(std::size_t left, std::size_t right) {
    if (right != 0 && left > std::numeric_limits<std::size_t>::max() / right) {
        throw std::length_error("buffer shape product overflows size_t");
    }
    return left * right;
}

inline std::uintmax_t negative_magnitude(std::ptrdiff_t value) noexcept {
    // Unlike -value, this also handles PTRDIFF_MIN without signed overflow.
    return static_cast<std::uintmax_t>(-(value + 1)) + 1;
}

inline std::ptrdiff_t checked_index_offset(std::size_t index, std::ptrdiff_t stride) {
    if (index == 0 || stride == 0) {
        return 0;
    }
    constexpr auto maximum = std::numeric_limits<std::ptrdiff_t>::max();
    if (stride > 0) {
        if (index > static_cast<std::uintmax_t>(maximum / stride)) {
            throw std::overflow_error("positive buffer byte offset overflows ptrdiff_t");
        }
        return static_cast<std::ptrdiff_t>(index) * stride;
    }
    const auto magnitude = negative_magnitude(stride);
    const auto negative_limit = static_cast<std::uintmax_t>(maximum) + 1;
    if (index > negative_limit / magnitude) {
        throw std::overflow_error("negative buffer byte offset overflows ptrdiff_t");
    }
    const auto product = static_cast<std::uintmax_t>(index) * magnitude;
    if (product == negative_limit) {
        return std::numeric_limits<std::ptrdiff_t>::min();
    }
    return -static_cast<std::ptrdiff_t>(product);
}

inline std::ptrdiff_t checked_add(std::ptrdiff_t left, std::ptrdiff_t right) {
    if ((right > 0 && left > std::numeric_limits<std::ptrdiff_t>::max() - right) ||
        (right < 0 && left < std::numeric_limits<std::ptrdiff_t>::min() - right)) {
        throw std::overflow_error("buffer byte offset sum overflows ptrdiff_t");
    }
    return left + right;
}

struct OffsetBounds {
    std::ptrdiff_t minimum = 0;
    std::ptrdiff_t maximum = 0;
};

inline void include_axis(OffsetBounds& bounds, std::size_t length, std::ptrdiff_t stride) {
    const auto endpoint = checked_index_offset(length - 1, stride);
    if (endpoint < 0) {
        bounds.minimum = checked_add(bounds.minimum, endpoint);
    } else {
        bounds.maximum = checked_add(bounds.maximum, endpoint);
    }
}

inline void validate_address_range(const void* data, OffsetBounds bounds) {
    if (data == nullptr) {
        throw std::invalid_argument("nonempty borrowed buffer has a null data pointer");
    }
    const auto before = bounds.minimum < 0 ? negative_magnitude(bounds.minimum) : 0;
    const auto after = static_cast<std::uintmax_t>(bounds.maximum);
    constexpr auto maximum_span = static_cast<std::uintmax_t>(
        std::numeric_limits<std::ptrdiff_t>::max());
    // Reject unrepresentable spans before adding/subtracting a pointer.
    if (before > maximum_span - sizeof(double) ||
        after > maximum_span - sizeof(double) - before) {
        throw std::overflow_error("borrowed buffer byte span exceeds PTRDIFF_MAX");
    }
    const auto address = reinterpret_cast<std::uintptr_t>(data);
    if (before > address || after + sizeof(double) - 1 >
            std::numeric_limits<std::uintptr_t>::max() - address) {
        throw std::overflow_error("borrowed buffer address range wraps uintptr_t");
    }
}

inline const void* checked_span_origin(const void* storage, std::size_t storage_bytes,
                                      std::size_t origin_bytes, OffsetBounds bounds,
                                      bool empty) {
    if (storage_bytes > static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max())) {
        throw std::length_error("storage byte span exceeds PTRDIFF_MAX");
    }
    if (origin_bytes > storage_bytes || (storage == nullptr && storage_bytes != 0)) {
        throw std::invalid_argument("invalid backing storage span or origin");
    }
    if (!empty) {
        const auto before = bounds.minimum < 0 ? negative_magnitude(bounds.minimum) : 0;
        const auto after = static_cast<std::uintmax_t>(bounds.maximum);
        if (before > origin_bytes || after > storage_bytes - origin_bytes ||
                sizeof(double) > storage_bytes - origin_bytes - after) {
            throw std::out_of_range("borrowed view reaches outside the supplied storage span");
        }
    }
    if (storage == nullptr) {
        return nullptr;  // Empty null spans never perform pointer arithmetic.
    }
    const auto address = reinterpret_cast<std::uintptr_t>(storage);
    if (storage_bytes != 0 && storage_bytes - 1 >
            std::numeric_limits<std::uintptr_t>::max() - address) {
        throw std::overflow_error("backing storage address range wraps uintptr_t");
    }
    return static_cast<const unsigned char*>(storage) + origin_bytes;
}

inline double read_double(const unsigned char* data, std::ptrdiff_t offset) noexcept {
    double value;
    // memcpy permits byte strides and logical origins that are not double-aligned.
    // The caller must keep the declared backing allocation alive.
    std::memcpy(&value, data + offset, sizeof(value));
    return value;
}

}  // namespace detail

// Borrowed views own only their shape/stride metadata. They neither retain nor
// release the backing allocation. A raw-pointer constructor requires all
// reachable element bytes to belong to a live backing allocation; arithmetic
// validation cannot establish allocation provenance or lifetime. from_span also
// checks the declared allocation bounds. No unchecked element accessor is exposed.
class VectorView {
public:
    VectorView() noexcept = default;

    VectorView(const void* data, std::size_t size, std::ptrdiff_t stride_bytes)
        : data_(static_cast<const unsigned char*>(data)),
          size_(detail::checked_count(size)), stride_(stride_bytes) {
        if (!empty()) {
            detail::OffsetBounds bounds;
            detail::include_axis(bounds, size_, stride_);
            detail::validate_address_range(data_, bounds);
        }
    }

    static VectorView from_span(const void* storage, std::size_t storage_bytes,
                                std::size_t origin_bytes, std::size_t size,
                                std::ptrdiff_t stride_bytes) {
        detail::checked_count(size);
        detail::OffsetBounds bounds;
        if (size != 0) {
            detail::include_axis(bounds, size, stride_bytes);
        }
        const auto* origin = detail::checked_span_origin(
            storage, storage_bytes, origin_bytes, bounds, size == 0);
        return VectorView(origin, size, stride_bytes);
    }

    std::size_t size() const noexcept { return size_; }
    std::ptrdiff_t stride_bytes() const noexcept { return stride_; }
    const void* data() const noexcept { return data_; }
    bool empty() const noexcept { return size_ == 0; }

    double get(std::size_t index) const {
        if (index >= size_) {
            throw std::out_of_range("vector index is outside its shape");
        }
        return detail::read_double(data_, detail::checked_index_offset(index, stride_));
    }

private:
    const unsigned char* data_ = nullptr;
    std::size_t size_ = 0;
    std::ptrdiff_t stride_ = 0;
};

class MatrixView {
public:
    MatrixView() noexcept = default;

    MatrixView(const void* data, std::size_t rows, std::size_t columns,
               std::ptrdiff_t row_stride_bytes, std::ptrdiff_t column_stride_bytes)
        : data_(static_cast<const unsigned char*>(data)), rows_(rows), columns_(columns),
          size_(detail::checked_count(detail::checked_product(rows, columns))),
          row_stride_(row_stride_bytes), column_stride_(column_stride_bytes) {
        if (!empty()) {
            detail::OffsetBounds bounds;
            detail::include_axis(bounds, rows_, row_stride_);
            detail::include_axis(bounds, columns_, column_stride_);
            detail::validate_address_range(data_, bounds);
        }
    }

    static MatrixView from_span(const void* storage, std::size_t storage_bytes,
                                std::size_t origin_bytes, std::size_t rows, std::size_t columns,
                                std::ptrdiff_t row_stride_bytes, std::ptrdiff_t column_stride_bytes) {
        const auto count = detail::checked_count(detail::checked_product(rows, columns));
        detail::OffsetBounds bounds;
        if (count != 0) {
            detail::include_axis(bounds, rows, row_stride_bytes);
            detail::include_axis(bounds, columns, column_stride_bytes);
        }
        const auto* origin = detail::checked_span_origin(
            storage, storage_bytes, origin_bytes, bounds, count == 0);
        return MatrixView(origin, rows, columns, row_stride_bytes, column_stride_bytes);
    }

    std::size_t rows() const noexcept { return rows_; }
    std::size_t columns() const noexcept { return columns_; }
    std::size_t size() const noexcept { return size_; }
    std::ptrdiff_t row_stride_bytes() const noexcept { return row_stride_; }
    std::ptrdiff_t column_stride_bytes() const noexcept { return column_stride_; }
    const void* data() const noexcept { return data_; }
    bool empty() const noexcept { return size_ == 0; }

    double get(std::size_t row, std::size_t column) const {
        if (row >= rows_ || column >= columns_) {
            throw std::out_of_range("matrix index is outside its shape");
        }
        const auto offset = detail::checked_add(
            detail::checked_index_offset(row, row_stride_),
            detail::checked_index_offset(column, column_stride_));
        return detail::read_double(data_, offset);
    }

private:
    const unsigned char* data_ = nullptr;
    std::size_t rows_ = 0;
    std::size_t columns_ = 0;
    std::size_t size_ = 0;
    std::ptrdiff_t row_stride_ = 0;
    std::ptrdiff_t column_stride_ = 0;
};

// A private, contiguous, zero-initialized allocation. No Python references or
// callbacks participate in ownership, moving, reset, or destruction.
class OwnedDoubleBuffer {
public:
    OwnedDoubleBuffer() noexcept = default;

    explicit OwnedDoubleBuffer(std::size_t size)
        : size_(detail::checked_count(size)),
          data_(size_ == 0 ? nullptr : new double[size_]()) {}

    OwnedDoubleBuffer(const OwnedDoubleBuffer&) = delete;
    OwnedDoubleBuffer& operator=(const OwnedDoubleBuffer&) = delete;

    OwnedDoubleBuffer(OwnedDoubleBuffer&& other) noexcept
        : size_(std::exchange(other.size_, 0)), data_(std::move(other.data_)) {}

    OwnedDoubleBuffer& operator=(OwnedDoubleBuffer&& other) noexcept {
        if (this != &other) {
            data_ = std::move(other.data_);
            size_ = std::exchange(other.size_, 0);
        }
        return *this;
    }

    ~OwnedDoubleBuffer() noexcept = default;

    static constexpr std::size_t max_size() noexcept {
        return static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max()) / sizeof(double);
    }
    std::size_t size() const noexcept { return size_; }
    std::size_t size_bytes() const noexcept { return size_ * sizeof(double); }
    bool empty() const noexcept { return size_ == 0; }
    double* data() noexcept { return data_.get(); }
    const double* data() const noexcept { return data_.get(); }

    double get(std::size_t index) const {
        if (index >= size_) {
            throw std::out_of_range("owned buffer index is outside its allocation");
        }
        return data_[index];
    }

    void set(std::size_t index, double value) {
        if (index >= size_) {
            throw std::out_of_range("owned buffer index is outside its allocation");
        }
        data_[index] = value;
    }

    void fill(double value) noexcept {
        for (std::size_t index = 0; index < size_; ++index) {
            data_[index] = value;
        }
    }

    void reset() noexcept {
        data_.reset();
        size_ = 0;
    }

    VectorView view() const {
        return VectorView(data(), size_, sizeof(double));
    }

    MatrixView matrix_view(std::size_t rows, std::size_t columns) const {
        if (detail::checked_product(rows, columns) != size_) {
            throw std::invalid_argument("matrix shape does not match owned buffer size");
        }
        const auto row_stride = size_ == 0 ? 0 :
            detail::checked_index_offset(columns, sizeof(double));
        return MatrixView(data(), rows, columns, row_stride, sizeof(double));
    }

private:
    // Declaration order initializes the checked count before allocation.
    std::size_t size_ = 0;
    std::unique_ptr<double[]> data_;
};

}  // namespace native
}  // namespace cyroot

#endif  // CYROOT_NATIVE_BUFFERS_HPP
