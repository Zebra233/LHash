#include <iostream>
#include <cstring>
#include <ctime>


using namespace std;

typedef unsigned char uchar;

//S-Box
static uint8_t S[16] = { 0xe,0x9,0xf,0x0,0xd,0x4,0xa,0xb,0x1,0x2,0x8,0x3,0x7,0x6,0xc,0x5 };

//置换
static uint8_t P96[12] = { 0x6,0x0,0x9,0xb,0x1,0x4,0xa,0x3,0x5,0x7,0x2,0x8 };
static uint8_t P128[16] = { 0x3,0x6,0x9,0xc,0x7,0xa,0xd,0x0,0xb,0xe,0x1,0x4,0xf,0x8,0x5,0x2 };
static uint8_t c = 120, r = 8; //建议的参数   内部置换参数-b
static uint8_t b = 128;   //内部置换参数-b
static uint8_t B = 128 / 4;
static uint8_t n = 128;   //摘要长度-n
uchar state[128 / 4];

//unsigned int C[18]={0x0012,0x0113,0x1301,0x3725,0x7E6C,0xECFE,0xC9DB,0x9280,0x2436,0x485A,0x8193,0x1200,0x2537
//        ,0x5A48,0xA5B7,0x5B49,0xB7A5,0x7F6D}; //轮常数

uint8_t C[72] = { 0x0,0x0,0x1,0x2,0x0,0x1,0x1,0x3,0x1,0x3,0x0,0x1,0x3,0x7,0x2,0x5,0x7,0xE,0x6,0xC,0xE,0xC,0xF,0xE,
               0xC,0x9,0xD,0xB,0x9,0x2,0x8,0x0,0x2,0x4,0x3,0x6,0x4,0x8,0x5,0xA,0x8,0x1,0x9,0x3,0x1,0x2,0x0,0x0,
               0x2,0x5,0x3,0x7,0x5,0xA,0x4,0x8,0xA,0x5,0xB,0x7,0x5,0xB,0x4,0x9,0xB,0x7,0xA,0x5,0x7,0xF,0x6,0xD };

//域上的*2 *4运算
uchar x2(uchar i) {
    uchar tmp = i & 0x0f;
    unsigned int flag = (tmp & 0x08); //是否需要模
    tmp <<= 1;
    if (flag)
        tmp ^= 0x13; //x^4+x^2+1
    tmp &= 0x0f;
    return tmp;
}
uchar x4(uchar i) {
    return x2(x2(i));
}


//void PBox() {
//    uchar stmp[32 / 2];
//    memset(stmp, 0, 32 / 2);
//    for (unsigned i = 0; i < 32 / 2; i++) {
//        stmp[i] = state[P128[i]];
//    }
//    memcpy(state, stmp, B / 2);
//}
//
//void SBox() {
//    for (unsigned i = 0; i < B / 2; i++)
//        state[i] = S[state[i]];
//}

void PBoxAndSBox() {
    uchar stmp[32 / 2];
    memset(stmp, 0, B / 2);
    for (unsigned i = 0; i < B / 2; i++) {
        stmp[i] = S[state[P128[i]]];
    }
    memcpy(state, stmp, B / 2);
}

void BTrans(uchar* X3, uchar* X2, uchar* X1, uchar* X0) {
    uchar tmp3 = *X3;
    uchar tmp2 = *X2;
    uchar tmp1 = *X1;
    *X3 = *X2;
    *X1 = *X0;
    *X2 = tmp1 ^ x2(*X0);
    *X0 = tmp3 ^ x4(tmp2);
}
void ATrans() {
    for (unsigned i = 0; i < 4; i++) {
        for (unsigned j = 0; j < B / 2; j += 4) {
            BTrans(&state[j], &state[j + 1], &state[j + 2], &state[j + 3]);
        }
    }
}



//Fb
void Fb() {
    for (int i = 0; i < 18; i++) {  //18轮迭代
        uchar tmp[32];
        memset(tmp, 0, B);
        for (unsigned j = B / 2; j < B; j++)
            tmp[j] = state[j - B / 2];

        state[0] ^= C[4 * i];
        state[1] ^= C[4 * i + 1];
        state[2] ^= C[4 * i + 2];
        state[3] ^= C[4 * i + 3];

        PBoxAndSBox();
        ATrans();

        for (unsigned j = 0; j < 32 / 2; j++)
            tmp[j] = state[j] ^ state[j + B / 2];

        memcpy(state, tmp, B);
    }
}

//uchar -> unit8_t
void LHash(uchar* input, uchar* output) {
    memset(state, 0, 128 / 4);  //初始化S0
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

void toNibble(uchar* input,uchar* output) {
    for (int i = 0; i < 64; i++) {
        output[2*i] = (input[i] & 0xf0) >>4;
        output[2*i+1] = input[i] & 0x0f;
    }
}

void test(int times) {

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
        for (int j = 0; j < times; j++)                          
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


    /*uchar msg[128] = { 0xFF,0xFE,0xFD,0xFC,0xFB,0xFA,0xF9,0xF8,0xF7,0xF6,0xF5,0xF4,
                    0xF3,0xF2,0xF1,0xF0,0xEF,0xEE,0xED,0xEC,0xEB,0xEA,0xE9,0xE8,
                    0xE7,0xE6,0xE5,0xE4,0xE3,0xE2,0xE1,0xE0 };*/
    uchar msg[128] = { 0xF,0xF,0xF,0xE,0xF,0xD,0xF,0xC,0xF,0xB,0xF,0xA,0xF,0x9,0xF,0x8,0xF,0x7,0xF,0x6,0xF,0x5,0xF,0x4,
                      0xF,0x3,0xF,0x2,0xF,0x1,0xF,0x0,0xE,0xF,0xE,0xE,0xE,0xD,0xE,0xC,0xE,0xB,0xE,0xA,0xE,0x9,0xE,0x8,
                      0xE,0x7,0xE,0x6,0xE,0x5,0xE,0x4,0xE,0x3,0xE,0x2,0xE,0x1,0xE,0x0 };
    /*uchar msg[128] = { 0 };
    uchar input[128] = "Hello,world!";
    toNibble(input,msg);*/

    //test(100);
    
    //验证正确性
    uchar output[128 / 4] = { 0 };
    LHash(msg, output);
    cout << "output: ";                        
    for (unsigned idx = 0; idx < n / 4; idx += 2)
        printf("%01X%01X ", output[idx], output[idx + 1]);
    cout << endl;

    return 0;
}
