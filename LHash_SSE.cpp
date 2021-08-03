#include <iostream>
#include <cstring>
#include <ctime>
#include <emmintrin.h> 
#include <tmmintrin.h>

#define _MM_SHUFFLE(fp3,fp2,fp1,fp0) (((fp3)<<6)|((fp2)<<4)|((fp1)<<2)|((fp0))) //pshufd 完成32位循环移位 的第二操作数
typedef unsigned char uchar;


using namespace std;



//static uint8_t P128[16] = { 0x3,0x6,0x9,0xc,0x7,0xa,0xd,0x0,0xb,0xe,0x1,0x4,0xf,0x8,0x5,0x2 };

static uint8_t c = 120, r = 8, r1 = 8; //建议的参数   内部置换参数-b
static uint8_t b = 128;   //内部置换参数-b
static uint8_t B = 128 / 4;
static uint8_t n = 128;   //摘要长度-n
uchar state[128 / 4];

uchar C[72] = { 0x0,0x0,0x1,0x2,0x0,0x1,0x1,0x3,0x1,0x3,0x0,0x1,0x3,0x7,0x2,0x5,0x7,0xE,0x6,0xC,0xE,0xC,0xF,0xE,
                  0xC,0x9,0xD,0xB,0x9,0x2,0x8,0x0,0x2,0x4,0x3,0x6,0x4,0x8,0x5,0xA,0x8,0x1,0x9,0x3,0x1,0x2,0x0,0x0,
                  0x2,0x5,0x3,0x7,0x5,0xA,0x4,0x8,0xA,0x5,0xB,0x7,0x5,0xB,0x4,0x9,0xB,0x7,0xA,0x5,0x7,0xF,0x6,0xD };


//__m128i P128 = { 0xC,0xD,0xE,0xF,0x9,0xA,0xB,0x2,0x6,0x7,0x4,0x5,0x3,0x0,0x1,0x8 };

//__m128i P128 = { 0x8,0x1,0x0,0x3,0x5,0x4,0x7,0x6,0x2,0xB,0xA,0x9,0xF,0xE,0xD,0xC };

//__m128i P128 = { 0x3,0x2,0x1,0x0,0x6,0x5,0x4,0x13,0x9,0x8,0x11,0x10,0x12,0x15,0x14,0x7 };

//__m128i P128 = { 0x7,0xE,0xF,0xC,0xA,0xB,0x8,0x9,0xD,0x4,0x5,0x6,0x0,0x1,0x2,0x3 };

__m128i Adj = { 0x0,0x4,0x8,0xC,0x1,0x5,0x9,0xD,0x2,0x6,0xA,0xE,0x3,0x7,0xB,0xF };

__m128i P128 = { 0x7,0xE,0xF,0xC,0xA,0xB,0x8,0x9,0xD,0x4,0x5,0x6,0x0,0x1,0x2,0x3 };

__m128i SBox = { 0xE,0x9,0xF,0x0,0xD,0x4,0xA,0xB,0x1,0x2,0x8,0x3,0x7,0x6,0xC,0x5 };

__m128i X2 = { 0x0,0x2,0x4,0x6,0x8,0xA,0xC,0xE,0x3,0x1,0x7,0x5,0xB,0x9,0xF,0xD };

__m128i X4 = { 0x0,0x4,0x8,0xC,0x3,0x7,0xB,0xF,0x6,0x2,0xE,0xA,0x5,0x1,0xD,0x9 };


void reverseState(uchar* rState) {

    uchar temp = 0x0;
    for (int i = 0; i < 128 / 8; i++) {
        temp = rState[i];
        rState[i] = rState[128 / 4 - 1 - i];
        rState[128 / 4 - 1 - i] = temp;
    }
}

void BTrans(__m128i* b3, __m128i b2, __m128i* b1, __m128i b0) {
    *b3 = _mm_xor_si128(*b3, _mm_shuffle_epi8(X4, b2));
    *b1 = _mm_xor_si128(*b1, _mm_shuffle_epi8(X2, b0));
}


void Fb() {
    //拆分
    __m128i X1 = _mm_setr_epi8(state[15], state[14], state[13], state[12], state[11], state[10], state[9], state[8], state[7],
        state[6], state[5], state[4], state[3], state[2], state[1], state[0]);
    __m128i X2 = _mm_setr_epi8(state[31], state[30], state[29], state[28], state[27], state[26], state[25], state[24], state[23],
        state[22], state[21], state[20], state[19], state[18], state[17], state[16]);
    __m128i X1Temp = X1;
    for (int i = 0; i < 18; i++) {  //18轮迭代
        X1Temp = X1;
        uchar rState[128 / 4];
        memcpy(rState, state, 128 / 4);

        __m128i CXor = _mm_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C[4 * i + 3], C[4 * i + 2], C[4 * i + 1], C[4 * i]);

        X1 = _mm_xor_si128(X1, CXor);    //轮常数异或

        X1 = _mm_shuffle_epi8(X1, Adj);  //X1调整字节顺序

        X1 = _mm_shuffle_epi8(X1, P128); //P盒  

        X1 = _mm_shuffle_epi8(SBox, X1); //S盒

        __m128i temp0, temp1, temp2;

        //拆分                               X1 B3
        temp2 = _mm_load_si128(&X1);
        temp2 = _mm_bslli_si128(temp2, 4);    //B2
        temp1 = _mm_load_si128(&X1);
        temp1 = _mm_bslli_si128(temp1, 8);    //B1
        temp0 = _mm_load_si128(&temp1);
        temp0 = _mm_bslli_si128(temp0, 4);    //B0

        //线性变换A
        BTrans(&X1, temp2, &temp1, temp0);
        BTrans(&temp2, temp1, &temp0, X1);
        BTrans(&temp1, temp0, &X1, temp2);
        BTrans(&temp0, X1, &temp2, temp1);




        //合并
        X1 = _mm_unpackhi_epi32(temp2, X1);
        temp1 = _mm_unpackhi_epi32(temp0, temp1);
        X1 = _mm_unpackhi_epi64(temp1, X1);


        X1 = _mm_shuffle_epi8(X1, Adj);  //还原字节顺序

        X1 = _mm_xor_si128(X1, X2);      //Xor

        X2 = X1Temp;

    }
    __m128 X1R = _mm_castsi128_ps(X1);
    __m128 X2R = _mm_castsi128_ps(X2);
    uchar stateTemp[16] = { 0 };  //保存原state前16nibble

    memcpy(state, &X2R, 16);
    memcpy(state + 16, &X1R, 16);


    reverseState(state);
}





void LHash(uchar* input, uchar* output) {
    memset(state, 0, 128 / 4);  //初始化S0;
    state[24] = 8; state[25] = 0;
    state[26] = 8; state[27] = 0;
    state[28] = 0; state[29] = 8;
    state[30] = 0; state[31] = 8;


    unsigned idx = 0;
    if (input[idx] != 0) {
        while (input[idx] != 0 || input[idx + 1] != 0) {
            state[0] ^= input[idx];
            state[1] ^= input[idx + 1];
            Fb();
            idx += 2;
        }
    }
    //填充
    if (input[idx] == 0 || input[idx + 1] == 0)  //正好r倍
        state[0] ^= 0x8;

    Fb();
    for (unsigned i = 0; i < n / 4; i += 2) {
        output[i] = state[0];
        output[i + 1] = state[1];

        Fb();
    }
}

void toNibble(uchar* input, uchar* output) {
    for (int i = 0; i < 64; i++) {
        output[2 * i] = (input[i] & 0xf0) >> 4;
        output[2 * i + 1] = input[i] & 0x0f;
    }
}


void test(int times) {                      //各长度输入n次LHash用时测试
    uchar msg_128B[258] = { 0 };
    for (int i = 0; i < 256; i++) {
        msg_128B[i] = 0xF;
    }


    uchar msg_256B[514] = { 0 };
    for (int i = 0; i < 512; i++) {
        msg_256B[i] = 0xF;
    }


    uchar msg_512B[1026] = { 0 };
    for (int i = 0; i < 1024; i++) {
        msg_512B[i] = 0xF;
    }

    uchar msg_1024B[2050] = { 0 };
    for (int i = 0; i < 2048; i++) {
        msg_1024B[i] = 0xF;
    }

    uchar output[128 / 4] = { 0 };

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
        printf("%01X%01X ", output[idx], output[idx + 1]);
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
        printf("%01X%01X ", output[idx], output[idx + 1]);
    cout << endl << endl;


    memset(cost, 0, 10);
    cout << "msg_512B:" << endl;
    for (int i = 0; i < 10; i++) {
        clock_t t1 = clock();
        for (int j = 0; j < times; j++)                         
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
        printf("%01X%01X ", output[idx], output[idx + 1]);
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
        printf("%01X%01X ", output[idx], output[idx + 1]);
    cout << endl << endl;
}


int main() {

    uchar msg[128] = { 0xF,0xF,0xF,0xE,0xF,0xD,0xF,0xC,0xF,0xB,0xF,0xA,0xF,0x9,0xF,0x8,0xF,0x7,0xF,0x6,0xF,0x5,0xF,0x4,
                      0xF,0x3,0xF,0x2,0xF,0x1,0xF,0x0,0xE,0xF,0xE,0xE,0xE,0xD,0xE,0xC,0xE,0xB,0xE,0xA,0xE,0x9,0xE,0x8,
                      0xE,0x7,0xE,0x6,0xE,0x5,0xE,0x4,0xE,0x3,0xE,0x2,0xE,0x1,0xE,0x0 };
    /*uchar msg[128] = { 0 };
    uchar input[128] = "Hello,world!";
    toNibble(input, msg);*/
    //test(1000);
    
    uchar output[128 / 4] = { 0 };
    LHash(msg, output);
    cout << "output: ";
    for (unsigned idx = 0; idx < n / 4; idx += 2)
        printf("%01X%01X ", output[idx], output[idx + 1]);
    cout << endl;

    return 0;
}


