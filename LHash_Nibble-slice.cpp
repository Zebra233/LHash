#include <iostream>
#include <cstring>
#include <ctime>
#include <emmintrin.h> 
#include <tmmintrin.h>

#define _MM_SHUFFLE(fp3,fp2,fp1,fp0)(((fp3)<<6)|((fp2)<<4)|((fp1)<<2)|((fp0)))

typedef unsigned char uchar;

using namespace std;

static uint8_t c = 120, r = 8, r1 = 8; //建议的参数   内部置换参数-b
static uint8_t b = 128;   //内部置换参数-b
static uint8_t B = 128 / 4;
static uint8_t n = 128;   //摘要长度-n
uchar state[4][128 / 4];


uint8_t C[72] = { 0x0,0x0,0x1,0x2,0x0,0x1,0x1,0x3,0x1,0x3,0x0,0x1,0x3,0x7,0x2,0x5,0x7,0xE,0x6,0xC,0xE,0xC,0xF,0xE,
                  0xC,0x9,0xD,0xB,0x9,0x2,0x8,0x0,0x2,0x4,0x3,0x6,0x4,0x8,0x5,0xA,0x8,0x1,0x9,0x3,0x1,0x2,0x0,0x0,
                  0x2,0x5,0x3,0x7,0x5,0xA,0x4,0x8,0xA,0x5,0xB,0x7,0x5,0xB,0x4,0x9,0xB,0x7,0xA,0x5,0x7,0xF,0x6,0xD };


__m128i Adj = { 0x0,0x4,0x8,0xC,0x1,0x5,0x9,0xD,0x2,0x6,0xA,0xE,0x3,0x7,0xB,0xF };

__m128i P128 = { 0x7,0xE,0xF,0xC,0xA,0xB,0x8,0x9,0xD,0x4,0x5,0x6,0x0,0x1,0x2,0x3 };

__m128i SBox = { 0xE,0x9,0xF,0x0,0xD,0x4,0xA,0xB,0x1,0x2,0x8,0x3,0x7,0x6,0xC,0x5 };

__m128i X2 = { 0x0,0x2,0x4,0x6,0x8,0xA,0xC,0xE,0x3,0x1,0x7,0x5,0xB,0x9,0xF,0xD };

__m128i X4 = { 0x0,0x4,0x8,0xC,0x3,0x7,0xB,0xF,0x6,0x2,0xE,0xA,0x5,0x1,0xD,0x9 };

__m128i mask2 = { 0xF,0xF, 0xF, 0xF, 0x0,0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0 };

__m128i mask1 = { 0x0,0x0, 0x0, 0x0, 0xF,0xF, 0xF, 0xF, 0xF,0xF, 0xF, 0xF, 0xF,0xF, 0xF, 0xF };





void reverse(uchar* rState, int len) {

    uchar temp = 0x0;
    for (int i = 0; i < len / 2; i++) {
        temp = rState[i];
        rState[i] = rState[len - 1 - i];
        rState[len - 1 - i] = temp;
    }
}


void BTrans(__m128i* b3, __m128i b2, __m128i* b1, __m128i b0) {
    *b3 = _mm_xor_si128(*b3, _mm_shuffle_epi8(X4, b2));
    *b1 = _mm_xor_si128(*b1, _mm_shuffle_epi8(X2, b0));
}

void Fb() {
    
    /*__m128i W1[4] = { _mm_setr_epi8(state[0][15], state[0][14], state[0][13], state[0][12], state[0][11], state[0][10], state[0][9], state[0][8], state[0][7],
        state[0][6], state[0][5], state[0][4], state[0][3], state[0][2], state[0][1], state[0][0]), 
        _mm_setr_epi8(state[1][15], state[1][14], state[1][13], state[1][12], state[1][11], state[1][10], state[1][9], state[1][8], state[1][7],
            state[1][6], state[1][5], state[1][4], state[1][3], state[1][2], state[1][1], state[1][0]),
        _mm_setr_epi8(state[2][15], state[2][14], state[2][13], state[2][12], state[2][11], state[2][10], state[2][9], state[2][8], state[2][7],
            state[2][6], state[2][5], state[2][4], state[2][3], state[2][2], state[2][1], state[2][0]),
        _mm_setr_epi8(state[3][15], state[3][14], state[3][13], state[3][12], state[3][11], state[3][10], state[3][9], state[3][8], state[3][7],
            state[3][6], state[3][5], state[3][4], state[3][3], state[3][2], state[3][1], state[3][0]) };

    __m128i W2[4] = { _mm_setr_epi8(state[0][31], state[0][30], state[0][29], state[0][28], state[0][27], state[0][26], state[0][25], state[0][24], state[0][23],
        state[0][22], state[0][21], state[0][20], state[0][19], state[0][18], state[0][17], state[0][16]), 
    _mm_setr_epi8(state[1][31], state[1][30], state[1][29], state[1][28], state[1][27], state[1][26], state[1][25], state[1][24], state[1][23],
            state[1][22], state[1][21], state[1][20], state[1][19], state[1][18], state[1][17], state[1][16]) ,
    _mm_setr_epi8(state[2][31], state[2][30], state[2][29], state[2][28], state[2][27], state[2][26], state[2][25], state[2][24], state[2][23],
            state[2][22], state[2][21], state[2][20], state[2][19], state[2][18], state[2][17], state[2][16]) ,
    _mm_setr_epi8(state[3][31], state[3][30], state[3][29], state[3][28], state[3][27], state[3][26], state[3][25], state[3][24], state[3][23],
            state[3][22], state[3][21], state[3][20], state[3][19], state[3][18], state[3][17], state[3][16]) };*/
    __m128i W1[4];
    __m128i W2[4];
    __m128i tempW1[4];

    __m128i test = _mm_setr_epi8(state[3][12], state[2][12], state[1][12], state[0][12], state[3][8], state[2][8], state[1][8], state[0][8], state[3][4],
        state[2][4], state[1][4], state[0][4], state[3][0], state[2][0], state[1][0], state[0][0]);


    W1[0] = _mm_setr_epi8(state[3][12], state[2][12], state[1][12], state[0][12], state[3][8], state[2][8], state[1][8], state[0][8], state[3][4],
        state[2][4], state[1][4], state[0][4], state[3][0], state[2][0], state[1][0], state[0][0]);
    W1[1] =_mm_setr_epi8(state[3][13], state[2][13], state[1][13], state[0][13], state[3][9], state[2][9], state[1][9], state[0][9], state[3][5],
        state[2][5], state[1][5], state[0][5], state[3][1], state[2][1], state[1][1], state[0][1]);
    W1[2] =_mm_setr_epi8(state[3][14], state[2][14], state[1][14], state[0][14], state[3][10], state[2][10], state[1][10], state[0][10], state[3][6],
        state[2][6], state[1][6], state[0][6], state[3][2], state[2][2], state[1][2], state[0][2]);
    W1[3] =_mm_setr_epi8(state[3][15], state[2][15], state[1][15], state[0][15], state[3][11], state[2][11], state[1][11], state[0][11], state[3][7],
        state[2][7], state[1][7], state[0][7], state[3][3], state[2][3], state[1][3], state[0][3]);


    W2[0]=_mm_setr_epi8(state[3][28], state[2][28], state[1][28], state[0][28], state[3][24], state[2][24], state[1][24], state[0][24], state[3][20],
        state[2][20], state[1][20], state[0][20], state[3][16], state[2][16], state[1][16], state[0][16]);
    W2[1]=_mm_setr_epi8(state[3][29], state[2][29], state[1][29], state[0][29], state[3][25], state[2][25], state[1][25], state[0][25], state[3][21],
            state[2][21], state[1][21], state[0][21], state[3][17], state[2][17], state[1][17], state[0][17]);
    W2[2]=_mm_setr_epi8(state[3][30], state[2][30], state[1][30], state[0][30], state[3][26], state[2][26], state[1][26], state[0][26], state[3][22],
            state[2][22], state[1][22], state[0][22], state[3][18], state[2][18], state[1][18], state[0][18]);
    W2[3]=_mm_setr_epi8(state[3][31], state[2][31], state[1][31], state[0][31], state[3][27], state[2][27], state[1][27], state[0][27], state[3][23],
            state[2][23], state[1][23], state[0][23], state[3][19], state[2][19], state[1][19], state[0][19]);


    for (int idx = 0; idx < 4; idx++)    //保存未异或轮常数的X1
         tempW1[idx] = W1[idx];
    for (int i = 0; i < 18; i++) {
        for (int idx = 0; idx < 4; idx++)
            tempW1[idx] = W1[idx];
        __m128i CXor[4] = { _mm_setr_epi8(0,0,0,0,0,0,0,0,0,0,0,0,C[4 * i], C[4 * i],C[4 * i],C[4 * i]) ,
        _mm_setr_epi8(0,0,0,0,0,0,0,0,0,0,0,0,C[4 * i + 1], C[4 * i + 1],C[4 * i + 1],C[4 * i + 1]) ,
        _mm_setr_epi8(0,0,0,0,0,0,0,0,0,0,0,0,C[4 * i + 2], C[4 * i + 2],C[4 * i + 2],C[4 * i + 2]),
        _mm_setr_epi8(0,0,0,0,0,0,0,0,0,0,0,0,C[4 * i + 3], C[4 * i + 3],C[4 * i + 3],C[4 * i + 3]) };
        for (int idx = 0; idx < 4; idx++) {
            W1[idx] = _mm_xor_si128(W1[idx], CXor[idx]);
        }



        //P128
        __m128i Wtemp0 = W1[0];
        __m128i Wtemp1 = W1[1];
        W1[0] = W1[3];
        W1[1] = _mm_shuffle_epi32(W1[2], _MM_SHUFFLE(2, 1, 0, 3));
        W1[2] = _mm_shuffle_epi32(Wtemp1, _MM_SHUFFLE(1, 0, 3, 2));
        W1[3] = _mm_shuffle_epi32(Wtemp0, _MM_SHUFFLE(0, 3, 2, 1));
        //n2 n8对换
        __m128i tempW11 = W1[1];
        W1[1] = _mm_or_si128(_mm_and_si128(tempW11, mask1), _mm_and_si128(W1[3], mask2));
        W1[3] = _mm_or_si128(_mm_and_si128(W1[3], mask1), _mm_and_si128(tempW11, mask2));



        //S盒
        for (int idx = 0; idx < 4; idx++) {
            W1[idx] = _mm_shuffle_epi8(SBox, W1[idx]);
        }

        //线性变换A
        BTrans(&W1[0], W1[1], &W1[2], W1[3]);
        BTrans(&W1[1], W1[2], &W1[3], W1[0]);
        BTrans(&W1[2], W1[3], &W1[0], W1[1]);
        BTrans(&W1[3], W1[0], &W1[1], W1[2]);


        for (int idx = 0; idx < 4; idx++) {
            W1[idx] = _mm_xor_si128(W1[idx], W2[idx]);
        }
        for (int idx = 0; idx < 4; idx++) {
            W2[idx] = tempW1[idx];
        }
    }





        __m128 W1R[4] = { _mm_castsi128_ps(W1[0]) ,_mm_castsi128_ps(W1[1]) , _mm_castsi128_ps(W1[2]) , _mm_castsi128_ps(W1[3]) };
        __m128 W2R[4] = { _mm_castsi128_ps(W2[0]) ,_mm_castsi128_ps(W2[1]) , _mm_castsi128_ps(W2[2]) , _mm_castsi128_ps(W2[3]) };
        uchar temp1[4][16] = { 0 };
        uchar temp2[4][16] = { 0 };
        for (int idx = 0; idx < 4; idx++) {
            memcpy(temp1[idx], &W1R[idx], 16);
            memcpy(temp2[idx], &W2R[idx], 16);
            reverse(temp1[idx], 16);
            reverse(temp2[idx], 16);
        }
        //还原
        uchar WChar1[4][16];
        uchar WChar2[4][16];
        for (int idx1 = 0; idx1 < 4; idx1++) {
            for (int idx2 = 0; idx2 < 16; idx2++) {
                WChar1[idx2 % 4][idx1 + 4 * (idx2 / 4)] = temp1[idx1][idx2];
                WChar2[idx2 % 4][idx1 + 4 * (idx2 / 4)] = temp2[idx1][idx2];
            }
        }
        //产生新state
        for (int idx = 0; idx < 4; idx++) {
            memcpy(state[idx] + 16,WChar2[idx], 16);
            memcpy(state[idx], WChar1[idx], 16);
        }
}




void LHash(uchar* input[], uchar* output[]) {
    for (int i = 0; i < 4; i++) {
        memset(state[i], 0, 128 / 4);  //初始化S0;
        state[i][24] = 8; state[i][25] = 0;
        state[i][26] = 8; state[i][27] = 0;
        state[i][28] = 0; state[i][29] = 8;
        state[i][30] = 0; state[i][31] = 8;
    }

    unsigned idx = 0;
    if (input[1][idx] != 0) {
        while (input[1][idx] != 0 || input[1][idx + 1] != 0) {
            for (int j = 0; j < 4; j++) {
                state[j][0] ^= input[j][idx];
                state[j][1] ^= input[j][idx + 1];
            }
            Fb();
            idx += 2;
        }
    }
    //填充
    if (input[0][idx] == 0 || input[0][idx + 1] == 0)  //正好r倍
        for (int j = 0; j < 4; j++)
            state[j][0] ^= 0x8;

    Fb();
    for (unsigned i = 0; i < n / 4; i += 2) {
        for (int q = 0; q < 4; q++) {
            output[q][i] = state[q][0];
            output[q][i + 1] = state[q][1];
        }
        Fb();
    }
}
void toNibble(uchar* input, uchar* output) {
    for (int i = 0; i < 64; i++) {
        output[2 * i] = (input[i] & 0xf0) >> 4;
        output[2 * i + 1] = input[i] & 0x0f;
    }
}

void test(int times){
    uchar msg128B[4][258];
    for (int i = 0; i < 4; i++) {
        memset(msg128B[i], 0xF, 256);
        msg128B[i][256] = 0x0;
        msg128B[i][257] = 0x0;
    }
    uchar* msg_128B[4] = { msg128B[0],msg128B[1], msg128B[2], msg128B[3] };

    uchar msg256B[4][514];
    for (int i = 0; i < 4; i++) {
        memset(msg256B[i], 0xF, 512);
        msg256B[i][512] = 0x0;
        msg256B[i][513] = 0x0;
    }
    uchar* msg_256B[4] = { msg256B[0],msg256B[1], msg256B[2], msg256B[3] };

    uchar msg512B[4][1026];
    for (int i = 0; i < 4; i++) {
        memset(msg512B[i], 0xF, 1024);
        msg512B[i][1024] = 0x0;
        msg512B[i][1025] = 0x0;
    }
    uchar* msg_512B[4] = { msg512B[0],msg512B[1], msg512B[2], msg512B[3] };

    uchar msg1024B[4][2050];
    for (int i = 0; i < 4; i++) {
        memset(msg1024B[i], 0xF, 2048);
        msg1024B[i][2048] = 0x0;
        msg1024B[i][2049] = 0x0;
    }
    uchar* msg_1024B[4] = { msg1024B[0],msg1024B[1], msg1024B[2], msg1024B[3] };



    uchar output1[128 / 4] = { 0 };
    uchar output2[128 / 4] = { 0 };
    uchar output3[128 / 4] = { 0 };
    uchar output4[128 / 4] = { 0 };
    uchar* output[4] = { output1,output2,output3,output4 };


    double cost[10] = { 0 };
    cout << "msg_128B:" << endl;
    for (int i = 0; i < 10; i++) {
        clock_t t1 = clock();
        for (int j = 0; j < times; j++)
            LHash(msg_128B, output);
        clock_t t2 = clock();
        cost[i] = double(t2 - t1) / CLOCKS_PER_SEC; //时间结果以秒为单位
        cout << cost[i] << " ";
    }
    double sum = 0;
    for (int i = 0; i < 10; i++)
        sum += cost[i];
    cout << "sum: " << sum << endl;
    cout << "output: ";
    for (unsigned idx = 0; idx < n / 4; idx += 2)
        printf("%01X%01X ", output[0][idx], output[0][idx + 1]);
    cout << endl << endl;




    memset(cost, 0, 10);
    cout << "msg_256B:" << endl;
    for (int i = 0; i < 10; i++) {
        clock_t t1 = clock();
        for (int j = 0; j < times; j++)
            LHash(msg_256B, output);
        clock_t t2 = clock();
        cost[i] = double(t2 - t1) / CLOCKS_PER_SEC; //时间结果以秒为单位
        cout << cost[i] << " ";
    }
    sum = 0;
    for (int i = 0; i < 10; i++)
        sum += cost[i];
    cout << "sum: " << sum << endl;
    cout << "output: ";
    for (unsigned idx = 0; idx < n / 4; idx += 2)
        printf("%01X%01X ", output[0][idx], output[0][idx + 1]);
    cout << endl << endl;


    memset(cost, 0, 10);
    cout << "msg_512B:" << endl;
    for (int i = 0; i < 10; i++) {
        clock_t t1 = clock();
        for (int j = 0; j < times; j++)                          //100次
            LHash(msg_512B, output);
        clock_t t2 = clock();
        cost[i] = double(t2 - t1) / CLOCKS_PER_SEC; //时间结果以秒为单位
        cout << cost[i] << " ";
    }
    sum = 0;
    for (int i = 0; i < 10; i++)
        sum += cost[i];
    cout << "sum: " << sum << endl;
    cout << "output: ";
    for (unsigned idx = 0; idx < n / 4; idx += 2)
        printf("%01X%01X ", output[0][idx], output[0][idx + 1]);
    cout << endl << endl;



    memset(cost, 0, 10);
    cout << "msg_1024B:" << endl;
    for (int i = 0; i < 10; i++) {
        clock_t t1 = clock();
        for (int j = 0; j < times; j++)                                  //100次
            LHash(msg_1024B, output);
        clock_t t2 = clock();
        cost[i] = double(t2 - t1) / CLOCKS_PER_SEC; //时间结果以秒为单位
        cout << cost[i] << " ";
    }
    sum = 0;
    for (int i = 0; i < 10; i++)
        sum += cost[i];
    cout << "sum: " << sum << endl;
    cout << "output: ";
    for (unsigned idx = 0; idx < n / 4; idx += 2)
        printf("%01X%01X ", output[0][idx], output[0][idx + 1]);
    cout << endl << endl;
}



int main() {

    uchar msg1[128] = { 0xF,0xF,0xF,0xE,0xF,0xD,0xF,0xC,0xF,0xB,0xF,0xA,0xF,0x9,0xF,0x8,0xF,0x7,0xF,0x6,0xF,0x5,0xF,0x4,
                      0xF,0x3,0xF,0x2,0xF,0x1,0xF,0x0,0xE,0xF,0xE,0xE,0xE,0xD,0xE,0xC,0xE,0xB,0xE,0xA,0xE,0x9,0xE,0x8,
                      0xE,0x7,0xE,0x6,0xE,0x5,0xE,0x4,0xE,0x3,0xE,0x2,0xE,0x1,0xE,0x0 };
    uchar msg2[128] = { 0xF,0xF,0xF,0xE,0xF,0xD,0xF,0xC,0xF,0xB,0xF,0xA,0xF,0x9,0xF,0x8,0xF,0x7,0xF,0x6,0xF,0x5,0xF,0x4,
                      0xF,0x3,0xF,0x2,0xF,0x1,0xF,0x0,0xE,0xF,0xE,0xE,0xE,0xD,0xE,0xC,0xE,0xB,0xE,0xA,0xE,0x9,0xE,0x8,
                      0xE,0x7,0xE,0x6,0xE,0x5,0xE,0x4,0xE,0x3,0xE,0x2,0xE,0x1,0xE,0x0 };
    uchar msg3[128] = { 0xF,0xF,0xF,0xE,0xF,0xD,0xF,0xC,0xF,0xB,0xF,0xA,0xF,0x9,0xF,0x8,0xF,0x7,0xF,0x6,0xF,0x5,0xF,0x4,
                      0xF,0x3,0xF,0x2,0xF,0x1,0xF,0x0,0xE,0xF,0xE,0xE,0xE,0xD,0xE,0xC,0xE,0xB,0xE,0xA,0xE,0x9,0xE,0x8,
                      0xE,0x7,0xE,0x6,0xE,0x5,0xE,0x4,0xE,0x3,0xE,0x2,0xE,0x1,0xE,0x0 };
    uchar msg4[128] = { 0xF,0xF,0xF,0xE,0xF,0xD,0xF,0xC,0xF,0xB,0xF,0xA,0xF,0x9,0xF,0x8,0xF,0x7,0xF,0x6,0xF,0x5,0xF,0x4,
                      0xF,0x3,0xF,0x2,0xF,0x1,0xF,0x0,0xE,0xF,0xE,0xE,0xE,0xD,0xE,0xC,0xE,0xB,0xE,0xA,0xE,0x9,0xE,0x8,
                      0xE,0x7,0xE,0x6,0xE,0x5,0xE,0x4,0xE,0x3,0xE,0x2,0xE,0x1,0xE,0x0 };
    uchar* msg[4] = { msg1,msg2,msg3,msg4 };

    /*uchar msg1[128] = { 0 }; uchar msg2[128] = { 0 }; uchar msg3[128] = { 0 }; uchar msg4[128] = { 0 };
    uchar input[128] = "Hello,world!";
    toNibble(input,msg1); toNibble(input, msg2); toNibble(input, msg3); toNibble(input, msg4);
    uchar* msg[4] = { msg1,msg2,msg3,msg4 };*/

    test(1000);

    /*uchar output1[128 / 4] = { 0 };
    uchar output2[128 / 4] = { 0 };
    uchar output3[128 / 4] = { 0 };
    uchar output4[128 / 4] = { 0 };
    uchar* output[4] = { output1,output2,output3,output4 };

    LHash(msg, output);

    for (int idx1 = 0; idx1 < 4; idx1++) {
        for (int idx2 = 0; idx2 < n / 4; idx2 += 2)
            printf("%01X%01X ", output[idx1][idx2], output[idx1][idx2+1]);
        cout << endl;
    }*/


    return 0;
}


