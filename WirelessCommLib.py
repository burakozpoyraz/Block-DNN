# ====================================================================================================|
# Burak Özpoyraz, 2021 																				  |
#                                                                                                     |
# Physical Layer of Wireless Communications Library                                                   |
#                                                                                                     |
# This code is the library of functions related to physical layer of wireless communications          |
# ====================================================================================================|

# LIBRARIES////////////////////////////////////////////////////////////////////////////////////////////
from collections import Counter
from copy import deepcopy
import numpy as np
import itertools
import math
# /////////////////////////////////////////////////////////////////////////////////////////////////////

# FUNCTIONS////////////////////////////////////////////////////////////////////////////////////////////
# =====================================================================================================
# 1. Binomial Combination
#
# ARGUMENTS
# 1-) n: Number of elements in the bigger set (Data Type: int)
# 2-) k: Number of elements in the sub set (Data Type: int)
#
# OUTPUT
# - N: Number of combinations (Data Type: float)
# =====================================================================================================
def Combination(n, k):
	N = math.factorial(n) / (math.factorial(n - k) * math.factorial(k))
	return N
# =====================================================================================================


# =====================================================================================================
# 2. Constellation
#
# DESCRIPTION: For QAM, only square lattice constellations are included.
#
# ARGUMENTS
# 1-) M: Constellation size (Data Type: int | Condition: Power of 2)
# 2-) mod_type: Modulation type (Data Type: str | Condition: "PSK" or "QAM")
# 3-) is_normalized: Whether each symbol in the constellation has unit power or not (Data Type: bool)
#
# OUTPUT
# - ss: Signal set (Data Type: numpy.ndarray | Shape: (1, M))
# =====================================================================================================
def Constellation(M, mod_type, is_normalized):
	pi = math.pi
	ss = np.zeros((1, M), dtype=np.complex64)
	if mod_type == "PSK":
		rad_angle_array = np.arange(0, 2 * pi, 2 * pi / M)
		data_index = 0
		for rad_angle in rad_angle_array:
			data_real = math.cos(rad_angle)
			data_imag = math.sin(rad_angle)
			ss[0][data_index] = data_real + 1j * data_imag
			data_index = data_index + 1
		ss = Bin2GrayCode(ss)
	elif mod_type == "QAM":
		data_real_array = Bin2GrayCode(np.arange(-(math.sqrt(M) - 1), math.sqrt(M), 2).reshape(1, -1)).reshape(-1,)
		data_imag_array = Bin2GrayCode(np.arange(-(math.sqrt(M) - 1), math.sqrt(M), 2).reshape(1, -1)).reshape(-1,)
		data_index = 0
		for data_real in data_real_array:
			for data_imag in data_imag_array:
				ss[0][data_index] = data_real + 1j * data_imag
				data_index = data_index + 1
		if is_normalized == True:
			Es = sum(sum(abs(ss) ** 2)) / M
			ss = ss / np.sqrt(Es)
	return ss
# =====================================================================================================


# =====================================================================================================
# 3. Conversion From Binary Coding To Gray Coding
#
# ARGUMENTS
# - bin_array: Binary coded array (Data Type: numpy.ndarray | Shape: (1, M))
#
# OUTPUT
# - gray_array: Gray coded array (Data Type: numpy.ndarray | Shape: (1, M))
# =====================================================================================================
def Bin2GrayCode(bin_array):
	length = bin_array.shape[1]
	bin_dec_array = np.arange(0, length)
	gray_array = np.zeros((1, length), dtype=np.complex64)
	dec_index = 0
	for dec in bin_dec_array:
		new_dec = dec ^ (dec >> 1)
		gray_array[0][dec_index] = bin_array[0][new_dec]
		dec_index = dec_index + 1
	return gray_array
# =====================================================================================================


# =====================================================================================================
# 4. Conversion From Decimal To Binary
#
# ARGUMENTS
# 1-) dec: Decimal number (Data Type: int)
# 2-) n: Number of desired bits at the output (Data Type: int)
#
# OUTPUT
# - bit_array: Bit array corresponding to the given decimal number (Data Type: numpy.ndarray | Shape:
# (1, M))
# =====================================================================================================
def Dec2Bin(dec, n):
	max_dec_with_n_bits = sum(2 ** np.arange(0, n))
	if dec <= max_dec_with_n_bits:
		new_dec = deepcopy(dec)
		decreasing_pow_array = np.arange(n-1, -1, -1)
		bit_array = np.zeros((1, n))
		bit_index = 0
		for pow_val in decreasing_pow_array:
			if (new_dec / (2 ** pow_val)) >= 1:
				bit_array[0][bit_index] = 1
				new_dec = new_dec - (2 ** pow_val)
			else:
				bit_array[0][bit_index] = 0
			bit_index = bit_index + 1
	return bit_array
# =====================================================================================================


# =====================================================================================================
# 5. Optimal Transmit Antenna Combination (TAC) Set
#
# DESCRIPTION: The aim of this function is to create a TAC set with a number of element that is a power
# of two, and that is smaller than the number of total antenna combinations. Optimality indicates that
# the TAC set is not created by randomly picking some of the TACs, instead TAC set is created so that 
# the number of each transmit antenna in the set is either equal or close to each other.
#
# ARGUMENTS
# 1-) Nt: Number of transmit antennas (Data Type: int)
# 2-) Np: Number of active transmit antennas (Data Type: int)
# 3-) N: Number of illegitimate TACs (Data Type: int)
#
# OUTPUT
# - TAC_set: Optimal TAC set (Data Type: numpy.ndarray | Shape: (N, Np))
# =====================================================================================================
def OptimalTAC_Set(Nt, Np, N):
	tx_ind_array = np.arange(1, Nt + 1)
	TAC_set = np.array(list(itertools.combinations(tx_ind_array, Np))) # All Nt choose Np combinations
	remaining_num_TAC = TAC_set.shape[0]
	while remaining_num_TAC > N:
		num_each_tx_ind_in_TAC_set_array = np.zeros((Nt,))
		for tx_ind in tx_ind_array:
			num_each_tx_ind_in_TAC_set_array[tx_ind - 1] = (TAC_set == tx_ind).sum()
		decreasing_sorted_index_array = np.argsort(-num_each_tx_ind_in_TAC_set_array)
		sorted_index_comb_set = np.array(list(itertools.combinations(decreasing_sorted_index_array, Np)))

		comb_index = 0
		is_comb_deleted = 0
		while is_comb_deleted == 0:
			comb = sorted(sorted_index_comb_set[comb_index] + 1)
			
			if DoesInclude(TAC_set, comb):
				row_indices, col_indices = np.where(TAC_set == comb)
				index_of_comb_in_TAC_set = SortByFreq(row_indices)[0]
				TAC_set = np.delete(TAC_set, index_of_comb_in_TAC_set, axis=0)
				is_comb_deleted = 1
			else:
				comb_index = comb_index + 1
		remaining_num_TAC = TAC_set.shape[0]
	return TAC_set
# =====================================================================================================


# =====================================================================================================
# 6. Sorting Array By Frequency
#
# ARGUMENT
# - array: Array to be sorted (Data Type: numpy.ndarray or list | Shape: (n,))
#
# OUTPUT
# - array_sorted_by_freq: Sorted array by frequency (Data Type: numpy.ndarray or list | Shape: (n,))
# =====================================================================================================
def SortByFreq(array):
	count_dict = Counter(array)
	array_sorted_by_freq = sorted(array, key=count_dict.get, reverse=True)
	return array_sorted_by_freq
# =====================================================================================================


# =====================================================================================================
# 7. Whether Matrix Includes Array In Any Of The Rows
#
# ARGUMENTS
# 1-) matrix: Matrix to be checked if any array is included in any of the row (Data Type: numpy.ndarray
# | Shape: (r, c))
# 2-) array: Array to be checked if exists in any row of the matrix (Data Type: numpy.ndarray or list |
#  Shape: (c,))
#
# OUTPUT
# - does_include: Whether matrix includes array in any of the rows (Data Type: bool)
# =====================================================================================================
def DoesInclude(matrix, array):
	num_row = matrix.shape[0]
	does_include = 0
	row_index = 0
	loop_cond = all([does_include == 0, row_index <= num_row])
	while loop_cond:
		row = matrix[row_index, :]
		if all(row == array):
			does_include = 1
		else:
			row_index = row_index + 1
		loop_cond = all([does_include == 0, row_index < num_row])
	return does_include
# =====================================================================================================


# =====================================================================================================
# 8. Encoding Bits Into Symbols
#
# ARGUMENTS
# 1-) bit_array: Bit array to be encoded (Data Type: numpy.ndarray or list | Shape: (n_tot,))
# 2-) ss: Signal set (Data Type: numpy.ndarray | Shape: (1, M))
# 3-) TAC_set: Optimal TAC set (Data Type: numpy.ndarray | Shape: (N, Np))
# 4-) ns: Number of spatial bits transmitted during a time-slot (Data Type: int)
# 5-) m: Number of information bits transmitted from a single active antenna during a time-slot (Data 
# Type: int)
# 6-) Nt: Number of transmit antennas (Data Type: int)
# 7-) Np: Number of active transmit antennas (Data Type: int)
#
# OUTPUT
# - x: Encoded symbol array (Data Type: numpy.ndarray | Shape: (Nt, 1))
# =====================================================================================================
def EncodeBits(bit_array, ss, TAC_set, ns, m, Nt, Np):
	x = np.zeros((Nt, 1), dtype=np.complex64)

	spatial_bits = bit_array[0 : ns]
	TAC_index = Bin2Dec(spatial_bits)
	TAC = TAC_set[TAC_index, :]

	for symbol_index in range(Np):
		start_bit_index = int(ns + symbol_index * m)
		stop_bit_index = int(ns + ((symbol_index + 1) * m))
		symbol_bits = bit_array[start_bit_index : stop_bit_index]
		symbol = ss[0][Bin2Dec(symbol_bits)]

		antenna_index = TAC[symbol_index] - 1
		x[antenna_index] = symbol
	return x
# =====================================================================================================


# =====================================================================================================
# 9. Conversion From Binary To Decimal
#
# ARGUMENT
# - bit_array: Bit array to be converted to decimal number (Data Type: numpy.ndarray or list | Shape: 
# (n,))
#
# OUTPUT
# - decimal_number: Decimal number corresponding to the bit_array (Data Type: int)
# =====================================================================================================
def Bin2Dec(bit_array):
	num_bits = len(bit_array)
	decimal_number = 0
	for bit_index in range(num_bits):
		bit = bit_array[num_bits - bit_index - 1]
		decimal_number = decimal_number + (bit * (2 ** bit_index))
	return decimal_number
# =====================================================================================================


# =====================================================================================================
# 10. Additional White Gaussian Noise
#
# ARGUMENTS
# 1-) N0: Noise power (Data Type: float)
# 2-) dim: Dimension of noise array (Data Type: numpy.ndarray or list | Shape: (2,))
#
# OUTPUT
# - n: Additional white Gaussian noise (Data Type: numpy.ndarray | Shape: (dim[0], dim[1]))
# =====================================================================================================
def AWGN(N0, dim):
	num_row = dim[0]
	num_col = dim[1]
	n = np.sqrt(N0 / 2) * (np.random.randn(num_row, num_col) + 1j * np.random.randn(num_row, num_col))
	return n
# =====================================================================================================


# =====================================================================================================
# 11. Rayleigh Fading Channel
#
# ARGUMENT
# - dim: Dimension of channel array (Data Type: numpy.ndarray or list | Shape: (2,))
#
# OUTPUT
# - H: Rayleigh fading channel (Data Type: numpy.ndarray | Shape: (dim[0], dim[1]))
# =====================================================================================================
def Channel(dim):
	num_row = dim[0]
	num_col = dim[1]
	H = np.sqrt(1 / 2) * (np.random.randn(num_row, num_col) + 1j * np.random.randn(num_row, num_col))
	return H
# =====================================================================================================


# =====================================================================================================
# 12. Feature Vector Generator
#
# ARGUMENTS
# 1-) complex_array: Complex array to be processed by FVG (Data Type: numpy.ndarray | Shape: (r, c))
# 2-) FVG_type: FVG Type (Data Type: str | Condition: "SFVG", "JFVG", or "CFVG")
#
# OUTPUT
# - fvg_output_array: Real vectorized array (Data Type: numpy.ndarray | Shape: (2 * r * c, 1))
# =====================================================================================================
def FVG(complex_array, FVG_type):
	if FVG_type == "SFVG":
		vectorized_complex_array = complex_array.reshape(-1, 1)
		fvg_output_array = np.concatenate((np.real(vectorized_complex_array), np.imag(vectorized_complex_array)))
	elif FVG_type == "JFVG":
		num_col = complex_array.shape[1]
		fvg_output_array = np.zeros((num_col ** 2, 1))
		element_index = 0
		for col_index1 in range(num_col):
			col1 = complex_array[:, col_index1]
			for col_index2 in range(num_col):
				col2 = complex_array[:, col_index2]
				fvg_output_array[element_index, 0] = abs(np.matmul(Hermitian(col1), col2))
	elif FVG_type == "CFVG":
		vectorized_complex_array = complex_array.reshape(-1, 1)
		fvg_output_array = abs(vectorized_complex_array)
	return fvg_output_array
# =====================================================================================================


# =====================================================================================================
# 13. Hermitian
#
# ARGUMENT
# - complex_array: Complex array (Data Type: numpy.ndarray | Shape: (r, c))
#
# OUTPUT
# - complex_array_herm: Hermitian of the input (Data Type: numpy.ndarray | Shape: (c, r))
# =====================================================================================================
def Hermitian(complex_array):
	complex_array_herm = np.transpose(np.conjugate(complex_array))
	return complex_array_herm
# =====================================================================================================
# /////////////////////////////////////////////////////////////////////////////////////////////////////