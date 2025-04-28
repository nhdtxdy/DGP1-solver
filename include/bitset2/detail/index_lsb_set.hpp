// BITSET2
//
//  Copyright Claas Bontus
//
//  Use, modification and distribution is subject to the
//  Boost Software License, Version 1.0. (See accompanying
//  file LICENSE.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
//
// Project home: https://github.com/ClaasBontus/bitset2
//


#ifndef BITSET2_INDEX_LSB_SET_CB_HPP
#define BITSET2_INDEX_LSB_SET_CB_HPP


#include <limits>
#include <climits>
#include <cstddef>
#include <type_traits>

#if __cplusplus >= 202002L
# include <bit>
# ifdef __cpp_lib_bitops
#  define CMPLRCOUNTR 1
# else
#  define CMPLRCOUNTR 0
# endif
#else
# define CMPLRCOUNTR 0
#endif



namespace Bitset2
{
namespace detail
{


/// https://graphics.stanford.edu/~seander/bithacks.html#ZerosOnRightBinSearch
template<class T>
struct index_lsb_set
{
  enum : size_t { npos=   std::numeric_limits<size_t>::max()
                , n_bits= sizeof(T) * CHAR_BIT };

  constexpr
  index_lsb_set() noexcept
  {
    static_assert( ( n_bits & ( n_bits - 1 ) ) == 0,
                   "Number of bits in data type is not a power of 2" );
  }
  
  constexpr
  size_t
  impl( T val ) const noexcept
  {
    size_t c = 1;
    size_t sh_rgt = n_bits >> 1;
    T pttrn = T( T(~T(0)) >> sh_rgt );
    
    while( sh_rgt >= 2 )
    {
        if( T( val & pttrn ) == T(0) ) 
        {
            c += sh_rgt;
            val >>= sh_rgt;
        }
        sh_rgt >>= 1;
        pttrn >>= sh_rgt;
    }
    if( T(val & T(1) ) == T(1) ) --c;
      
    return c;
  } // impl
  
#ifdef __cpp_lib_bitops
  constexpr
  size_t
  impl2( T val ) const noexcept
  { 
# ifdef __SIZEOF_INT128__
    if constexpr( std::is_same_v<T,unsigned __int128> ) return impl(val);
    else
# endif
    return std::countr_zero(val);
  }
#endif

  /// \brief Returns index of first (least significant) bit set in val.
  /// Returns npos if all bits are zero.
  constexpr
  size_t
  operator()( T val ) const noexcept
  {
    if( T(0) == val ) return npos;
    if constexpr( CMPLRCOUNTR ) return impl2(val);
    return impl(val);
  }
}; // struct index_lsb_set



} // namespace detail
} // namespace Bitset2


#undef CMPLRCOUNTR

#endif // BITSET2_INDEX_LSB_SET_CB_HPP
