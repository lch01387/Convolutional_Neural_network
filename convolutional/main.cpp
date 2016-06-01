// Convolutional neural network

// 개선점1 : 불필요한 반복
// 개선점2 : 가중치 배열 순서바꾸기
// output layer에 sigmoid 썼는지 확인하기

// error를 그냥 맨뒷쪽에 몰아놓으면 되기때문에..

// memory waste: weight[0][0][0~3][5][5]
//              c_layer[0][0][~][~]

#include <iostream>
#include <fstream>
#include <math.h>
#include <time.h>
#include <random> 
#include <omp.h>

#define W_SIZE 5 // subsampling model size
#define HIDDEN 3 // number of(input layer + hidden layer)
#define FN 1, 4, 8 // number of FeatureMap of input layer + hidden layer.
#define OUTPUT 10 // output layer
#define PULL_NUM 2 // max pulling number
#define BATCH_SIZE 1 // size of mini batch

using namespace std;

void setweight();
void set_mnist();
void read_train();
void check_mnist();
void forwardprop();
void errorprop();
void backprop();
void read_test();
void close_train();
void close_test();
void open_train();

int reverseInt (int i);

int c_num[HIDDEN] = {FN};  // number of FeatureMaps
int* c_size; // size of FeatureMap's row&col
float** i_layer; // input layer
float**** c_layer; // subsampled layer
float**** m_layer; // max layer
float* f_layer; // fully connected layer
float***** weight;
float** f_weight; // f_weight // fully-connected layer weight
float o_layer[OUTPUT]; // output layer
float targets[OUTPUT];

float o_error[OUTPUT];
float* f_error;
float**** c_error;
float**** m_error;

int n_rows;
int n_cols; // size of input layer
int magic_number;
int number_of_images;
int magic_number1;
int number_of_images1;

int target;
int f_size; // size of fully connected layer

static float rate = 0.15; // learning rate
// rate 0.01 -> 85.94%
// rate 0.05 -> 91.21% x2  -> 93.18% x10-> 96.05%
// rate 0.10 -> 93.60% x10 -> 96.32%
// rate 0.13 -> 94.03% x10 -> 95.76%
//// 4*16 94.93
// rate 0.15 -> 94.06% x10 -> 95.39%
//// 4*16 95.32
// rate 0.16 -> 93.92%
// rate 0.17 -> 93.38%
// rate 0.20 -> 92.75%
// rate 0.30 -> 91.10%
int hit = 0;

ifstream file_train_images;
ifstream file_train_label;
ifstream file_test_images;
ifstream file_test_label;


int main(){
    time_t current_time;
    open_train();
    set_mnist();
    close_train();
    setweight(); // randomly set weight(weight, f_weight)
    for(int k=0; k<1; k++){
        cout << k+1 << "st" << endl;
        open_train();
        for(int r=0; r<number_of_images/BATCH_SIZE; r++){
            if(r%1000 == 0)
                cout << "progress: " << float(r)/700 << "%" << endl;

            #pragma omp parallel for num_threads(BATCH_SIZE)
            for(int m=0; m<BATCH_SIZE; m++){
                read_train(); // read once, i_layer is changed to new one
                forwardprop();
                errorprop();
            }
            backprop();
        }
        close_train();
    }
    read_test();
    close_test();
    cout << "인식율 : " << float(hit)/100 << "%" << endl;
    time( &current_time);
    printf(ctime( &current_time));
}

void set_mnist(){
    // 초기설정

    c_size = new int[HIDDEN];

    weight = new float****[HIDDEN-1];
    for(int i=0; i<HIDDEN-1; i++){
        weight[i] = new float***[c_num[i]];
        for(int j=0; j<c_num[i]; j++){
            weight[i][j] = new float**[c_num[i+1]];
            for(int k=0; k<c_num[i+1]; k++){
                weight[i][j][k] = new float*[W_SIZE];
                for(int z=0; z<W_SIZE; z++)
                    weight[i][j][k][z] = new float[W_SIZE];
            }
        }
    }

    c_size[0] = n_rows; // input layer
    c_size[1] = c_size[0] - W_SIZE + 1;
    for(int i=2; i<HIDDEN; i++)
        c_size[i] = (c_size[i-1] / PULL_NUM) - W_SIZE + 1;

    i_layer = new float*[n_rows];
    for(int i=0; i<n_rows; i++)
        i_layer[i] = new float[n_cols];

    c_layer = new float***[HIDDEN];
    c_error = new float***[HIDDEN];
    for(int i=0; i<HIDDEN; i++){
        c_layer[i] = new float**[c_num[i]];
        c_error[i] = new float**[c_num[i]];
        for(int j=0; j<c_num[i]; j++){
            c_layer[i][j] = new float*[c_size[i]];
            c_error[i][j] = new float*[c_size[i]];
            for(int k=0; k<c_size[i]; k++){
                c_layer[i][j][k] = new float[c_size[i]];
                c_error[i][j][k] = new float[c_size[i]];
            }
        }
    }

    m_layer = new float***[HIDDEN];
    m_error = new float***[HIDDEN];
    for(int i=0; i<HIDDEN; i++){
        m_layer[i] = new float**[c_num[i]];
        m_error[i] = new float**[c_num[i]];
        for(int j=0; j<c_num[i]; j++){
            m_layer[i][j] = new float*[c_size[i]/PULL_NUM];
            m_error[i][j] = new float*[c_size[i]/PULL_NUM];
            for(int k=0; k<c_size[i]/PULL_NUM; k++){
                m_layer[i][j][k] = new float[c_size[i]/PULL_NUM];
                m_error[i][j][k] = new float[c_size[i]/PULL_NUM];
            }
        }
    }

    f_size = c_size[HIDDEN-1]/PULL_NUM * c_size[HIDDEN-1]/PULL_NUM * c_num[HIDDEN-1];

    f_weight = new float*[OUTPUT];
    for(int i=0; i<OUTPUT; i++)
        f_weight[i] = new float[f_size];

    f_layer = new float[f_size];
    f_error = new float[f_size];

    close_train();
}

void read_train()
{
    if (file_train_images.is_open()){
        for(int r=0;r<n_rows;++r){
            for(int c=0;c<n_cols;++c){
                unsigned char temp=0;
                file_train_images.read((char*)&temp,sizeof(temp));
                i_layer[r][c] = float(temp)/128-1;
            }
        }
    }
    if (file_train_label.is_open()){
        unsigned char temp=0;
        file_train_label.read((char*)&temp,sizeof(temp));
        target = temp;
    }
    // set target matrix
    for(int o=0; o<OUTPUT; o++){
        if(o==target)
            targets[o] = 1;
        else
            targets[o] = 0;
    }
}

void forwardprop(){
    for(int h=0; h<HIDDEN; h++){
        // subsampling
        if(h==0){
            // input layer-> convolutional layer
            for(int fn=0; fn<c_num[h+1]; fn++)
                for(int fr=0; fr<c_size[h+1]; fr++) // row of convolutional layer
                    for(int fc=0; fc<c_size[h+1]; fc++){ // col of convolutional layer
                        float tempsum = 0;
                        for(int fb=0; fb<c_num[h]; fb++)
                            for(int wr=0; wr<W_SIZE; wr++) // row of weight
                                for(int wc=0; wc<W_SIZE; wc++){ // col of weight
                                    tempsum += i_layer[fr+wr][fc+wc] * weight[h][fb][fn][wr][wc];
                                }
                        c_layer[h+1][fn][fr][fc] = 1/(1+exp(-tempsum));
                    }
        }else if(h == HIDDEN-1){
            // convolutional layer -> fully connected layer
            int index = 0;
            for(int fn=0; fn<c_num[h]; fn++)
                for(int r=0; r<c_size[h]/PULL_NUM; r++) // row of last max-pooling layer
                    for(int c=0; c<c_size[h]/PULL_NUM; c++){ // col of last max-pooling layer
                        f_layer[index] = m_layer[h][fn][r][c];
                        index++;
                    }
            continue;
        }else{
            // convolutional layer -> convolutional layer
            for(int fn=0; fn<c_num[h+1]; fn++) // next c_num
                for(int fr=0; fr<c_size[h+1]; fr++) // row of next convolutional layer
                    for(int fc=0; fc<c_size[h+1]; fc++){ // col of next convolutional layer
                        float tempsum = 0;
                        for(int fb=0; fb<c_num[h]; fb++){ // back c_num
                            for(int wr=0; wr<W_SIZE; wr++)
                                for(int wc=0; wc<W_SIZE; wc++)
                                    tempsum += m_layer[h][fb][fr+wr][fc+wc] * weight[h][fb][fn][wr][wc];
                        }
                        c_layer[h+1][fn][fr][fc] = 1/(1+exp(-tempsum));
                    }
        }
        // max-pooling
        for(int f=0; f<c_num[h+1]; f++){
            for(int mr=0; mr<c_size[h+1]/PULL_NUM; mr++) // row of max-pooling layer
                for(int mc=0; mc<c_size[h+1]/PULL_NUM; mc++){ // col of max-pooling layer
                    float tempmax=-1; // -1 is minimum result of sigmoid func
                    for(int r=0; r<PULL_NUM; r++) // max-pooling size = 2
                        for(int c=0; c<PULL_NUM; c++){
                            if(c_layer[h+1][f][mr*PULL_NUM+r][mc*PULL_NUM+c]>tempmax)
                                tempmax = c_layer[h+1][f][mr*PULL_NUM+r][mc*PULL_NUM+c];
                        }
                    m_layer[h+1][f][mr][mc] = tempmax;
                }
        }
    }
    // fully connected layer -> output layer
    for(int i=0; i<OUTPUT; i++){
        float tempsum = 0;
        for(int j=0; j<f_size; j++)
            tempsum += f_weight[i][j]*f_layer[j];
        o_layer[i] = 1/(1+exp(-tempsum));
    }
    // calculate result
    int result = 0;
    float temp = -1;
    for(int i=0; i<OUTPUT; i++)
        if(o_layer[i] > temp){
            result= i;
            temp = o_layer[i];
        }
}

void setweight(){ // randomly set weight
    time_t   current_time;
    time( &current_time);
    printf(ctime( &current_time));
    int randn;

    // weight
    for(int h=0; h<HIDDEN-1; h++)
        for(int fb=0; fb<c_num[h]; fb++)
            for(int fn=0; fn<c_num[h+1]; fn++)
                for(int wr=0; wr<W_SIZE; wr++)
                    for(int wc=0; wc<W_SIZE; wc++){
                        randn = rand();
                        weight[h][fb][fn][wr][wc] = (randn%1000-500) * 0.001;
                    }

    // f_weight
    for(int o=0; o<OUTPUT; o++)
        for(int fs=0; fs<f_size; fs++){
            randn = rand();
            f_weight[o][fs] = (randn%1000-500) * 0.001;
        }
}

void errorprop(){
    // output layer -> fully connected layer
    for(int o=0; o<OUTPUT; o++)
        o_error[o] = (o_layer[o]-targets[o])*o_layer[o]*(1-o_layer[o]);
    
    // error of fully connected layer
    for(int o=0; o<OUTPUT; o++){
        for(int f=0; f<f_size; f++){
            f_error[f] += o_error[o]*f_weight[o][f] * (1-f_layer[f])*f_layer[f];
        }
    }
    // error of last max-pooling layer
    int index = 0;
    for(int fn=0; fn<c_num[HIDDEN-1]; fn++)
        for(int r=0; r<c_size[HIDDEN-1]/PULL_NUM; r++) // row of last max-pooling layer
            for(int c=0; c<c_size[HIDDEN-1]/PULL_NUM; c++){ // col of last max-pooling layer
                m_error[HIDDEN-1][fn][r][c] = f_error[index];
                index++;
            }
    
    
    for(int h=HIDDEN-1; h>1; h--){
        // max pooling layer -> convolution layer
        for(int f=0; f<c_num[h]; f++)
            for(int mr=0; mr<c_size[h]/PULL_NUM; mr++) // row of max-pooling layer
                for(int mc=0; mc<c_size[h]/PULL_NUM; mc++) // col of max-pooling layer
                    for(int r=0; r<PULL_NUM; r++) // max-pooling size = 2
                        for(int c=0; c<PULL_NUM; c++)
                            c_error[h][f][mr*PULL_NUM+r][mc*PULL_NUM+c] = m_error[h][f][mr][mc];
        
        // error of previous max-pooling layer
        for(int fn=0; fn<c_num[h]; fn++) // next c_num
            for(int fr=0; fr<c_size[h]; fr++) // row of next convolutional layer
                for(int fc=0; fc<c_size[h]; fc++) // col of next convolutional layer
                    for(int fb=0; fb<c_num[h-1]; fb++){ // back c_num
                        for(int wr=0; wr<W_SIZE; wr++)
                            for(int wc=0; wc<W_SIZE; wc++){
                                m_error[h-1][fb][fr+wr][fc+wc] += c_error[h][fn][fr][fc] * weight[h-1][fb][fn][wr][wc] * m_layer[h-1][fb][fr+wr][fc+wc]*(1 - m_layer[h-1][fb][fr+wr][fc+wc]);
                            }
                    }
    }
}

void backprop(){
    // output layer -> fully connected layer
    for(int o=0; o<OUTPUT; o++){
        for(int f=0; f<f_size; f++){
            // update fully connected layer weight
            f_weight[o][f] = f_weight[o][f] - rate*(o_error[o]*f_layer[f]);
        }
    }
    for(int f=0; f<f_size; f++)
        f_error[f] = 0;
    for(int h=HIDDEN-1; h>1; h--){
        // colvolution layer -> previous max pooling layer
        for(int fn=0; fn<c_num[h]; fn++) // next c_num
            for(int fr=0; fr<c_size[h]; fr++) // row of next convolutional layer
                for(int fc=0; fc<c_size[h]; fc++) // col of next convolutional layer
                    for(int fb=0; fb<c_num[h-1]; fb++){ // back c_num
                        for(int wr=0; wr<W_SIZE; wr++)
                            for(int wc=0; wc<W_SIZE; wc++){
                                // update convolutional layer weight
                                weight[h-1][fb][fn][wr][wc] = weight[h-1][fb][fn][wr][wc] - rate*(c_error[h][fn][fr][fc] * m_layer[h-1][fb][fr+wr][fc+wc]);
                            }
                    }
        for(int fb=0; fb<c_num[h-1]; fb++)
            for(int fr=0; fr<c_size[h-1]/PULL_NUM; fr++)
                for(int fc=0; fc<c_size[h-1]/PULL_NUM; fc++)
                    m_error[h-1][fb][fr][fc] = 0;
    }
    
    // convolution layer -> input layer
    if(HIDDEN > 1){
        for(int fn=0; fn<c_num[1]; fn++) // next c_num
            for(int fr=0; fr<c_size[1]; fr++) // row of next convolutional layer
                for(int fc=0; fc<c_size[1]; fc++) // col of next convolutional layer
                    for(int fb=0; fb<c_num[0]; fb++){ // back c_num
                        for(int wr=0; wr<W_SIZE; wr++)
                            for(int wc=0; wc<W_SIZE; wc++){
                                // update input layer weight
                                weight[0][fb][fn][wr][wc] = weight[0][fb][fn][wr][wc] - rate*(c_error[1][fn][fr][fc] * i_layer[fr+wr][fc+wc]);
                            }
                    }
    }
}
void open_train(){
    file_train_images.open("/Users/changhyeon/Desktop/3-1/UROP/machine/machine/train-images-idx3-ubyte");
    file_train_label.open("/Users/changhyeon/Desktop/3-1/UROP/machine/machine/train-labels.idx1-ubyte");
    if (file_train_images.is_open())
    {
        file_train_images.read((char*)&magic_number,sizeof(magic_number));
        magic_number= reverseInt(magic_number);
        file_train_images.read((char*)&number_of_images,sizeof(number_of_images));
        number_of_images= reverseInt(number_of_images);
        file_train_images.read((char*)&n_rows,sizeof(n_rows));
        n_rows= reverseInt(n_rows);
        file_train_images.read((char*)&n_cols,sizeof(n_cols));
        n_cols= reverseInt(n_cols);

        file_train_label.read((char*)&magic_number1,sizeof(magic_number1));
        magic_number1= reverseInt(magic_number1);
        file_train_label.read((char*)&number_of_images1,sizeof(number_of_images1));
        number_of_images1= reverseInt(number_of_images1);
    }
}
void close_train(){
    file_train_images.close();
    file_train_label.close();
}
void close_test(){
    file_test_images.close();
    file_test_label.close();
}

int reverseInt (int i)
{
    unsigned char c1, c2, c3, c4;

    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;

    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

void read_test(){
    file_test_images.open("/Users/changhyeon/Desktop/3-1/UROP/machine/machine/t10k-images-idx3-ubyte");
    file_test_label.open("/Users/changhyeon/Desktop/3-1/UROP/machine/machine/t10k-labels.idx1-ubyte");
    if (file_test_images.is_open())
    {
        file_test_images.read((char*)&magic_number,sizeof(magic_number));
        magic_number= reverseInt(magic_number);
        file_test_images.read((char*)&number_of_images,sizeof(number_of_images));
        number_of_images= reverseInt(number_of_images);
        file_test_images.read((char*)&n_rows,sizeof(n_rows));
        n_rows= reverseInt(n_rows);
        file_test_images.read((char*)&n_cols,sizeof(n_cols));
        n_cols= reverseInt(n_cols);

        file_test_label.read((char*)&magic_number1,sizeof(magic_number1));
        magic_number1= reverseInt(magic_number1);
        file_test_label.read((char*)&number_of_images1,sizeof(number_of_images1));
        number_of_images1= reverseInt(number_of_images1);
    }

    for(int r=0; r<number_of_images; r++){
        if (file_test_images.is_open())
        {
            for(int r=0;r<n_rows;++r){
                for(int c=0;c<n_cols;++c){
                    unsigned char temp=0;
                    file_test_images.read((char*)&temp,sizeof(temp));
                    i_layer[r][c] = float(temp)/128-1;
                }
            }
        }
        if (file_test_label.is_open()){
            unsigned char temp=0;
            file_test_label.read((char*)&temp,sizeof(temp));
            target = temp;
        }
        // set target matrix
        for(int o=0; o<OUTPUT; o++){
            if(o==target)
                targets[o] = 1;
            else
                targets[o] = 0;
        }

        if(r%1000 == 0)
            cout << "progress: " << float(r+60000)/700 << "%" << endl;
        for(int h=0; h<HIDDEN; h++){
            // subsampling
            if(h==0){
                // input layer-> convolutional layer
                for(int fn=0; fn<c_num[h+1]; fn++)
                    for(int fr=0; fr<c_size[h+1]; fr++) // row of convolutional layer
                        for(int fc=0; fc<c_size[h+1]; fc++){ // col of convolutional layer
                            float tempsum = 0;
                            for(int fb=0; fb<c_num[h]; fb++)
                                for(int wr=0; wr<W_SIZE; wr++) // row of weight
                                    for(int wc=0; wc<W_SIZE; wc++){ // col of weight
                                        tempsum += i_layer[fr+wr][fc+wc] * weight[h][fb][fn][wr][wc];
                                    }
                            c_layer[h+1][fn][fr][fc] = 1/(1+exp(-tempsum));
                        }
            }else if(h == HIDDEN-1){
                // convolutional layer -> fully connected layer
                int index = 0;
                for(int fn=0; fn<c_num[h]; fn++)
                    for(int r=0; r<c_size[h]/PULL_NUM; r++) // row of last max-pooling layer
                        for(int c=0; c<c_size[h]/PULL_NUM; c++){ // col of last max-pooling layer
                            f_layer[index] = m_layer[h][fn][r][c];
                            index++;
                        }
                continue;
            }else{
                // convolutional layer -> convolutional layer
                for(int fn=0; fn<c_num[h+1]; fn++) // next c_num
                    for(int fr=0; fr<c_size[h+1]; fr++) // row of next convolutional layer
                        for(int fc=0; fc<c_size[h+1]; fc++){ // col of next convolutional layer
                            float tempsum = 0;
                            for(int fb=0; fb<c_num[h]; fb++){ // back c_num
                                for(int wr=0; wr<W_SIZE; wr++)
                                    for(int wc=0; wc<W_SIZE; wc++)
                                        tempsum += m_layer[h][fb][fr+wr][fc+wc] * weight[h][fb][fn][wr][wc];
                            }
                            c_layer[h+1][fn][fr][fc] = 1/(1+exp(-tempsum));
                        }
            }
            // max-pooling
            for(int f=0; f<c_num[h+1]; f++){
                for(int mr=0; mr<c_size[h+1]/PULL_NUM; mr++) // row of max-pooling layer
                    for(int mc=0; mc<c_size[h+1]/PULL_NUM; mc++){ // col of max-pooling layer
                        float tempmax=-1; // -1 is minimum result of sigmoid func
                        for(int r=0; r<PULL_NUM; r++) // max-pooling size = 2
                            for(int c=0; c<PULL_NUM; c++){
                                if(c_layer[h+1][f][mr*PULL_NUM+r][mc*PULL_NUM+c]>tempmax)
                                    tempmax = c_layer[h+1][f][mr*PULL_NUM+r][mc*PULL_NUM+c];
                            }
                        m_layer[h+1][f][mr][mc] = tempmax;
                    }
            }
        }
        // fully connected layer -> output layer
        for(int i=0; i<OUTPUT; i++){
            float tempsum = 0;
            for(int j=0; j<f_size; j++)
                tempsum += f_weight[i][j]*f_layer[j];
            o_layer[i] = 1/(1+exp(-tempsum));
        }

        // calculate result
        int result = 0;
        float temp = -1;
        for(int i=0; i<OUTPUT; i++)
            if(o_layer[i] > temp){
                result= i;
                temp = o_layer[i];
            }
        // this image represent the number 'result'
        if(result == target)
            hit++;
        }
}
