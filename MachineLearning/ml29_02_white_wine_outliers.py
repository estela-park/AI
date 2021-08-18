# To determine whether outliers are the reason for low performance

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def outliers_by_feature(data_out):
    outliers = {}
    for i in range(data_out.shape[1]):
        lower_quartile, mid, upper_quartile = np.percentile(data_out[:, i], [25, 50, 75])
        iqr = upper_quartile - lower_quartile
        print('***********Quartile Index***********')
        print('lowest:', np.percentile(data_out[:, i], 0))
        print('Q1:', lower_quartile)
        print('Q2:', mid)
        print('Q3:', upper_quartile)
        print('highest:', np.percentile(data_out[:, i], 100))
        lower_bound = lower_quartile - iqr*1.5
        upper_bound = upper_quartile + iqr*1.5
        outliers[i] = (np.where((data_out[:, i] > upper_bound) | (data_out[:, i] < lower_bound)), sum(1 for x in data_out[:, i] if (x > upper_bound) | (x < lower_bound)))
    return  outliers


data = pd.read_csv('../_data/white_wine.csv', sep=';', index_col=None, header=0)
data = data.iloc[:, :11].values
print(outliers_by_feature(data))

'''
{0: ((array([  98,  169,  207,  294,  358,  551,  555,  656,  774,  847,  873,
       1053, 1109, 1123, 1124, 1138, 1139, 1141, 1142, 1146, 1147, 1178,
       1205, 1210, 1214, 1228, 1239, 1263, 1300, 1307, 1308, 1309, 1312,
       1313, 1334, 1349, 1372, 1373, 1404, 1420, 1423, 1505, 1526, 1536,
       1544, 1561, 1564, 1580, 1581, 1586, 1621, 1624, 1626, 1627, 1690,
       1718, 1730, 1758, 1790, 1801, 1856, 1857, 1858, 1900, 1930, 1932,
       1936, 1951, 1961, 2014, 2017, 2028, 2030, 2050, 2083, 2127, 2154,
       2162, 2191, 2206, 2250, 2266, 2308, 2312, 2321, 2357, 2378, 2400,
       2401, 2404, 2535, 2540, 2541, 2542, 2607, 2625, 2639, 2668, 2872,
       3094, 3095, 3220, 3265, 3307, 3410, 3414, 3526, 3710, 3915, 4259,
       4446, 4470, 4518, 4522, 4679, 4786, 4787, 4792, 4847], dtype=int64),), 119), 1: ((array([  17,   20,   23,   79,  147,  178,  188,  202,  208,  221,  224,
        230,  237,  267,  268,  269,  271,  273,  294,  311,  372,  392,
        407,  433,  443,  450,  478,  506,  508,  564,  626,  662,  687,
        766,  817,  819,  821,  926,  948, 1007, 1027, 1029, 1034, 1040,
       1042, 1057, 1074, 1114, 1119, 1125, 1152, 1171, 1180, 1217, 1245,
       1249, 1271, 1273, 1304, 1339, 1341, 1342, 1350, 1369, 1382, 1386,
       1394, 1401, 1417, 1436, 1453, 1470, 1476, 1477, 1496, 1499, 1541,
       1577, 1596, 1636, 1640, 1708, 1732, 1759, 1775, 1817, 1831, 1833,
       1835, 1848, 1856, 1886, 1931, 1932, 1951, 1970, 2014, 2017, 2092,
       2128, 2154, 2162, 2359, 2394, 2408, 2417, 2450, 2451, 2475, 2517,
       2531, 2532, 2546, 2589, 2594, 2595, 2629, 2651, 2668, 2731, 2732,
       2738, 2741, 2781, 3022, 3069, 3097, 3165, 3417, 3454, 3528, 3560,
       3564, 3571, 3635, 3655, 3659, 3662, 3665, 3677, 3710, 3773, 3879,
       3901, 3962, 4039, 4065, 4092, 4099, 4136, 4262, 4263, 4316, 4433,
       4479, 4484, 4501, 4503, 4552, 4596, 4597, 4609, 4617, 4619, 4622,
       4625, 4636, 4638, 4639, 4648, 4649, 4650, 4680, 4686, 4701, 4702,
       4779, 4789, 4792, 4815, 4836, 4847, 4860, 4867, 4877, 4878],
      dtype=int64),), 186), 2: ((array([  14,   16,   54,   62,   65,   84,   85,   86,   88,   89,   90,        
         96,   99,  115,  120,  137,  141,  158,  159,  178,  207,  230,
        238,  267,  268,  269,  271,  281,  296,  300,  302,  315,  402,
        433,  439,  464,  465,  468,  470,  496,  499,  501,  528,  541,
        549,  556,  570,  580,  593,  594,  602,  603,  612,  613,  614,
        615,  624,  646,  681,  694,  700,  745,  763,  766,  772,  780,
        792,  800,  805,  811,  853,  862,  864,  870,  871,  890,  913,
        915,  921,  922,  929,  937,  946,  969,  970,  979,  980,  999,
       1007, 1024, 1051, 1085, 1090, 1104, 1152, 1153, 1185, 1245, 1254,
       1256, 1261, 1262, 1282, 1304, 1307, 1308, 1310, 1326, 1332, 1415,
       1418, 1419, 1423, 1436, 1440, 1445, 1455, 1457, 1458, 1460, 1465,
       1487, 1488, 1489, 1504, 1507, 1511, 1525, 1530, 1534, 1540, 1551,
       1560, 1563, 1569, 1570, 1575, 1578, 1583, 1584, 1585, 1587, 1588,
       1589, 1590, 1592, 1596, 1598, 1604, 1635, 1651, 1722, 1732, 1744,
       1768, 1775, 1817, 1846, 1879, 1881, 1885, 1886, 1896, 1897, 1905,
       1925, 1933, 1985, 2000, 2001, 2025, 2044, 2092, 2128, 2139, 2148,
       2173, 2178, 2186, 2226, 2227, 2317, 2318, 2321, 2322, 2346, 2371,
       2385, 2417, 2465, 2466, 2475, 2556, 2559, 2563, 2629, 2634, 2637,
       2721, 2781, 2806, 2807, 2820, 2848, 2856, 2866, 3043, 3052, 3064,
       3066, 3152, 3186, 3275, 3454, 3497, 3519, 3571, 3587, 3588, 3589,
       3616, 3625, 3635, 3650, 3677, 3709, 3710, 3737, 3807, 3808, 3848,
       3911, 3972, 4060, 4061, 4173, 4180, 4259, 4263, 4298, 4344, 4346,
       4430, 4503, 4565, 4567, 4591, 4597, 4609, 4626, 4632, 4649, 4650,
       4680, 4686, 4698, 4701, 4702, 4730, 4751, 4755, 4779, 4780, 4792,
       4806, 4808, 4815, 4847, 4877, 4878], dtype=int64),), 270), 3: ((array([1608, 1653, 1663, 2781, 3619, 3623, 4480], dtype=int64),), 7), 4: ((array([  23,   35,   40,   41,   54,   60,  110,  124,  194,  195,  196,        
        251,  315,  366,  433,  465,  478,  484,  506,  525,  531,  600,
        620,  621,  626,  662,  683,  687,  729,  754,  766,  771,  772,
        775,  814,  859,  870,  877,  878,  979,  980, 1024, 1034, 1051,
       1057, 1059, 1062, 1063, 1064, 1140, 1158, 1163, 1192, 1198, 1217,
       1254, 1272, 1278, 1323, 1369, 1551, 1598, 1599, 1638, 1651, 1672,
       1673, 1714, 1728, 1744, 1798, 1802, 1835, 1836, 1839, 1865, 1925,
       1926, 1933, 1937, 1965, 1972, 1973, 1974, 2024, 2025, 2026, 2120,
       2123, 2162, 2186, 2242, 2257, 2259, 2279, 2286, 2287, 2349, 2359,
       2365, 2371, 2379, 2380, 2392, 2393, 2412, 2414, 2422, 2424, 2440,
       2476, 2486, 2489, 2491, 2492, 2522, 2525, 2559, 2563, 2578, 2579,
       2631, 2644, 2649, 2654, 2704, 2705, 2755, 2781, 2785, 2787, 2820,
       2849, 2905, 2922, 2949, 2962, 3043, 3049, 3051, 3215, 3218, 3220,
       3283, 3288, 3388, 3434, 3537, 3627, 3628, 3629, 3638, 3678, 3686,
       3694, 3699, 3700, 3708, 3735, 3737, 3770, 3773, 3797, 3806, 3831,
       3832, 3848, 3873, 3901, 3902, 3911, 3937, 3961, 3972, 4063, 4089,
       4093, 4095, 4123, 4173, 4189, 4213, 4247, 4299, 4300, 4301, 4316,
       4344, 4346, 4349, 4370, 4473, 4480, 4497, 4555, 4614, 4648, 4698,
       4717, 4775, 4776, 4793, 4794, 4811, 4813, 4820, 4836, 4845],
      dtype=int64),), 208), 5: ((array([  67,  297,  325,  387,  395,  405,  459,  659,  752,  766, 1257,        
       1674, 1688, 1759, 1842, 1855, 1859, 1931, 2334, 2336, 2575, 2625,
       2728, 2735, 2748, 2750, 2872, 2893, 2930, 3050, 3072, 3307, 3379,
       3387, 3461, 3470, 3520, 3523, 3620, 3861, 3862, 3863, 3868, 3869,
       3871, 3981, 4179, 4185, 4745, 4841], dtype=int64),), 50), 6: ((array([ 227,  325,  387,  740, 1417, 1931, 
1940, 1942, 2127, 2378, 2654,
       3050, 3094, 3095, 3152, 3710, 3901, 4514, 4745], dtype=int64),), 19), 7: ((array([1653, 1663, 2781, 3619, 
3623], dtype=int64),), 5), 8: ((array([  72,  115,  250,  320,  507,  509,  830,  834,  892,  928, 1014,
       1095, 1214, 1250, 1255, 1335, 1352, 1361, 1385, 1482, 1575, 1578,
       1583, 1649, 1681, 1758, 1834, 1852, 1900, 1946, 1959, 1960, 1976,
       2036, 2063, 2075, 2078, 2099, 2104, 2162, 2211, 2238, 2247, 2280,
       2281, 2319, 2321, 2364, 2369, 2370, 2399, 2646, 2711, 2771, 2853,
       2862, 2864, 2872, 2895, 2956, 2964, 3025, 3128, 3556, 3598, 3762,
       4109, 4135, 4259, 4470, 4565, 4567, 4601, 4744, 4787], dtype=int64),), 75), 9: ((array([  80,  154,  209, 
 245,  339,  357,  411,  415,  530,  563,  701,
        757,  758,  759,  778,  782,  797,  852,  854,  855,  866,  868,
        879,  974, 1016, 1036, 1099, 1160, 1169, 1280, 1285, 1293, 1294,
       1386, 1394, 1407, 1412, 1455, 1482, 1515, 1590, 1807, 1809, 1843,
       1848, 1862, 1969, 1971, 1995, 1997, 1998, 2006, 2057, 2073, 2211,
       2234, 2264, 2267, 2348, 2403, 2441, 2594, 2634, 2637, 2642, 2656,
       2668, 2721, 2748, 2750, 2872, 2873, 2874, 2893, 2926, 2930, 2931,
       3057, 3079, 3206, 3207, 3231, 3423, 3425, 3426, 3429, 3430, 3431,
       3436, 3458, 3532, 3641, 3642, 3680, 3683, 3685, 3697, 3736, 3754,
       3764, 3904, 3915, 3975, 3982, 3998, 3999, 4000, 4012, 4023, 4026,
       4065, 4072, 4239, 4391, 4401, 4582, 4617, 4696, 4753, 4792, 4815,
       4818, 4886, 4887], dtype=int64),), 124), 10: ((array([], dtype=int64),), 0)}
'''