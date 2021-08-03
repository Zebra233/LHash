# 轻量级Hash算法-LHash

LHash-128/128/8/8

## 基本实现

[LHash.cpp](./LHash.cpp)

## 优化实现

1. [基于SSE指令的优化实现](./LHash_SSE.cpp)
2. [基于Nibble- slice技术的优化实现](./LHash_Nibble-slice.cpp)

## 摘要值

| Message             | FF FE FD FC FB FA F9 F8 F7 F6 F5 F4 F3 F2 F1 F0 EF EE ED EC EB EA E9 E8 E7 E6 E5 E4 E3 E2 E1 E0 |
| ------------------- | ------------------------------------------------------------ |
| LHash-80/96/16/16   | 4A BD BA E1 44 7F C8 E4 5B 58                                |
| LHash-96/96/16/16   | 55 EC 4F FE 99 2A 32 94 F1 F7 90 61                          |
| LHash-128/128/16/32 | 38 E9 1A E1 8F 11 5A 0B 27 79 68 22 A9 0B 1C 5A              |
| LHash-128/128/8/8   | 0A D6 35 B4 8F E3 BD 84 F9 58 7C 68 B0 CA DA E0              |

