#include <cstdio>
#include <vector>
#include <fstream>
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <Windows.h>

#include <conio.h>
#include <graphics.h>
#include<easyx.h>
#include<time.h>

using namespace std;

// -------------------------------------神经网络----------------------------------

vector< vector<double> > train_images;        // 训练集图像
vector<double> train_labels;                  // 训练集标签
vector< vector<double> > test_images;         // 测试集图像
vector<double> test_labels;                   // 测试集标签
vector<vector<double>>strengthen_images; //增强数据集图像
vector<double>strengthen_labels;	//增强数据集标签

const double learning_rate = 0.08;      // 学习率
const int epoch = 15;                   // 学习轮次
const int nh = 30;                      // 隐藏单元数量（单隐藏层）
double w1[784][nh];                     // 输入层到隐藏层的权重矩阵
double w2[nh][10];                      // 隐藏层到输出层的权重矩阵
double bias1[nh];                       // 隐藏层的偏置
double bias2[10];                       // 输出层的偏置

// -------------------------------------卷积神经网络----------------------------------

const double learning_speed = 0.08;
const int epoch2 = 15;						//学习轮次
const int layer_number = 30;					//卷积层层数
const int convolution_size = 25;				//卷积核边长
const int size1 = 28 - convolution_size + 1;//卷积层边长
const int press_rate = 2;					//压缩率
const int size2 = size1 / press_rate;			//池化层边长

double weight1[layer_number][convolution_size][convolution_size];//卷积核
double layer_bias[layer_number];								//卷积层偏置
double weight2[10][layer_number][size2][size2];					//池化层输出权重
double pool_bias[10];											//池化层输出偏置

int e;
// -------------------------------------函数声明----------------------------------

//神经网络
void test();
void write_parameters();
void read_parameters();
void train_and_save_parameters();

//卷积神经网络
void test2();
void write_parameters2();
void read_parameters2();
void train_and_save_parameters2();

//数据增强
int RotationRight90(vector<double> src);  //顺时针转90°
int RotationLeft90(vector<double>  src);	//逆时针转90°
int RotationDown(vector<double>  src);	//顺时针转180°
int Filp1(vector<double>  src);		//左右翻转
int Filp2(vector<double>  src);		//上下翻转
int geometryTrans1(vector<double>  src);//平移1
int geometryTrans2(vector<double>  src);//平移2


void fun();
void show(int index);
void mosaic(DWORD* pMem, int t, double(*p)[28]);//对此次手图像进行马赛克处理，并获取每个区域的亮度，用二维数组保存
void button_1(int x1, int y1, int x2, int y2);//画出按钮
int recognize(double(*p)[28]);
void progress_bar(int x1, int y1, int x2, int y2, double progress);
void current_button(int function);
void data_strengthen();
// -------------------------------------函数定义----------------------------------

int reverse_int(int i) {
	unsigned char ch1, ch2, ch3, ch4;
	ch1 = i & 255;
	ch2 = (i >> 8) & 255;
	ch3 = (i >> 16) & 255;
	ch4 = (i >> 24) & 255;
	return((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}
// 顺时针90度
int RotationRight90(vector<double> src)
{
	int srcH = 28;
	int srcW = 28;
	int channel = 1;
	vector<double> tempSrc;
	int mSize = srcW * srcH * sizeof(char) * channel;
	int i = 0;
	int j = 0;
	int k = 0;
	int desW = 0;
	int desH = 0;

	desW = srcH;
	desH = srcW;


	for (i = 0; i < desH; i++)
	{
		for (j = 0; j < desW; j++)
		{
			for (k = 0; k < channel; k++)
			{
				tempSrc.push_back(src[((srcH - 1 - j) * srcW + i) * channel + k]);
			}

		}
	}
	strengthen_images.push_back(tempSrc);
	return 0;
}


// 逆时针90度
int RotationLeft90(vector<double>  src)
{
	int srcH = 28;
	int srcW = 28;
	int channel = 1;
	vector<double> tempSrc;
	int mSize = srcW * srcH * sizeof(char) * channel;
	int i = 0;
	int j = 0;
	int k = 0;
	int desW = 0;
	int desH = 0;

	desW = srcH;
	desH = srcW;

	for (i = 0; i < desH; i++)
	{
		for (j = 0; j < desW; j++)
		{
			for (k = 0; k < channel; k++)
			{
				tempSrc.push_back(src[(j * srcW + i) * channel + k]);
			}

		}
	}
	strengthen_images.push_back(tempSrc);
	return 0;
}

// 旋转180度
int RotationDown(vector<double>  src)
{
	int srcH = 28;
	int srcW = 28;
	int channel = 1;
	vector<double> tempSrc;
	int mSize = srcW * srcH * sizeof(char) * channel;
	int i = 0;
	int j = 0;
	int k = 0;
	int desW = 0;
	int desH = 0;

	desW = srcW;
	desH = srcH;


	for (i = 0; i < desH; i++)
	{
		for (j = 0; j < desW; j++)
		{
			for (k = 0; k < channel; k++)
			{
				tempSrc.push_back(src[((srcH - 1 - i) * srcW + srcW - 1 - j) * channel + k]);
			}

		}
	}
	strengthen_images.push_back(tempSrc);
	return 0;
}

//左右翻转
int Filp1(vector<double>  src)
{
	int srcH = 28;
	int srcW = 28;
	int channel = 1;
	vector<double> tempSrc;
	int mSize = srcW * srcH * sizeof(char) * channel;
	int i = 0;
	int j = 0;
	int k = 0;
	int desW = 0;
	int desH = 0;

	desW = srcW;
	desH = srcH;


	for (i = 0; i < desH; i++)
	{
		for (j = 0; j < desW; j++)
		{
			for (k = 0; k < channel; k++)
			{
				tempSrc.push_back(src[(i * desW + srcH - j) * channel + k]);
			}

		}
	}
	strengthen_images.push_back(tempSrc);
	return 0;
}

//上下翻转
int Filp2(vector<double>  src)
{
	int srcH = 28;
	int srcW = 28;
	int channel = 1;
	vector<double> tempSrc;
	int mSize = srcW * srcH * sizeof(char) * channel;
	int i = 0;
	int j = 0;
	int k = 0;
	int desW = 0;
	int desH = 0;

	desW = srcW;
	desH = srcH;

	for (i = 0; i < desH; i++)
	{
		for (j = 0; j < desW; j++)
		{
			for (k = 0; k < channel; k++)
			{
				tempSrc.push_back(src[((desW - i) * desW + j) * channel + k]);
			}

		}
	}
	strengthen_images.push_back(tempSrc);
	return 0;
}

//平移
int geometryTrans1(vector<double>  src)
{
	int i, j;
	int intCapX, intCapY;
	int intXOffset = 1; //水平偏移量
	int intYOffset = 0; //垂直偏移量
	vector<double> tempSrc;
	for (i = 0; i < 28; i++)
	{
		for (j = 0; j < 28; j++)
		{
			intCapX = j - intXOffset;
			intCapY = i - intYOffset;
			// 判断 是否在原图范围内
			if ((intCapX >= 0) && (intCapX < 28) && (intCapY >= 0) && (intCapY < 28))
			{
				tempSrc.push_back(src[intCapX * 28 + intCapY]);
			}
			else
			{
				tempSrc.push_back(255);
			}
		}
	}
	strengthen_images.push_back(tempSrc);
	return 0;
}


//平移
int geometryTrans2(vector<double>  src)
{
	int i, j;
	int intCapX, intCapY;
	int intXOffset = -1; //水平偏移量
	int intYOffset = 0; //垂直偏移量
	vector<double> tempSrc;
	for (i = 0; i < 28; i++)
	{
		for (j = 0; j < 28; j++)
		{
			intCapX = j - intXOffset;
			intCapY = i - intYOffset;
			// 判断 是否在原图范围内
			if ((intCapX >= 0) && (intCapX < 28) && (intCapY >= 0) && (intCapY < 28))
			{
				tempSrc.push_back(src[intCapX * 28 + intCapY]);
			}
			else
			{
				tempSrc.push_back(255);
			}
		}
	}
	strengthen_images.push_back(tempSrc);
	return 0;
}

// 载入训练集图像
void read_train_images() {
	ifstream file("train-images-idx3-ubyte", ios::binary);
	if (file.is_open()) {
		int magic_number = 0;
		int number_of_images = 0;
		int row = 0;
		int col = 0;

		file.read((char*)&magic_number, sizeof(magic_number));
		file.read((char*)&number_of_images, sizeof(number_of_images));
		file.read((char*)&row, sizeof(row));
		file.read((char*)&col, sizeof(col));

		magic_number = reverse_int(magic_number);
		number_of_images = reverse_int(number_of_images);
		row = reverse_int(row);
		col = reverse_int(col);

		for (int i = 0; i < number_of_images; i++) {
			vector<double> this_image;
			for (int r = 0; r < row; r++) {
				for (int c = 0; c < col; c++) {
					unsigned char pixel = 0;
					file.read((char*)&pixel, sizeof(pixel));
					this_image.push_back(pixel);
					this_image[r * 28 + c] /= 255;          // 像素值归一化处理
				}
			}
			train_images.push_back(this_image);
		}
		printf("%d, train images success\n", train_images.size());
	}
}
// 载入训练集标签
void read_train_labels() {
	ifstream file;
	file.open("train-labels-idx1-ubyte", ios::binary);
	if (file.is_open()) {
		int magic_number = 0;
		int number_of_images = 0;

		file.read((char*)&magic_number, sizeof(magic_number));
		file.read((char*)&number_of_images, sizeof(number_of_images));

		magic_number = reverse_int(magic_number);
		number_of_images = reverse_int(number_of_images);

		for (int i = 0; i < number_of_images; i++) {
			unsigned char label = 0;
			file.read((char*)&label, sizeof(label));
			train_labels.push_back((double)label);
		}
		printf("%d, train labels success\n", train_labels.size());
	}
}
// 载入测试集图像
void read_test_images() {
	ifstream file("t10k-images-idx3-ubyte", ios::binary);
	if (file.is_open()) {
		int magic_number = 0;
		int number_of_images = 0;
		int row = 0;
		int col = 0;

		file.read((char*)&magic_number, sizeof(magic_number));
		file.read((char*)&number_of_images, sizeof(number_of_images));
		file.read((char*)&row, sizeof(row));
		file.read((char*)&col, sizeof(col));

		magic_number = reverse_int(magic_number);
		number_of_images = reverse_int(number_of_images);
		row = reverse_int(row);
		col = reverse_int(col);

		for (int i = 0; i < number_of_images; i++) {
			vector<double> this_image;
			for (int r = 0; r < row; r++) {
				for (int c = 0; c < col; c++) {
					unsigned char pixel = 0;
					file.read((char*)&pixel, sizeof(pixel));
					this_image.push_back(pixel);
					this_image[r * 28 + c] /= 255;          // 像素值归一化处理
				}
			}
			test_images.push_back(this_image);
		}
		printf("read %d test images success.\n", test_images.size());
	}
}
// 载入测试集标签
void read_test_labels() {
	ifstream file;
	file.open("t10k-labels-idx1-ubyte", ios::binary);
	if (file.is_open()) {
		int magic_number = 0;
		int number_of_images = 0;

		file.read((char*)&magic_number, sizeof(magic_number));
		file.read((char*)&number_of_images, sizeof(number_of_images));

		magic_number = reverse_int(magic_number);
		number_of_images = reverse_int(number_of_images);

		for (int i = 0; i < number_of_images; i++) {
			unsigned char label = 0;
			file.read((char*)&label, sizeof(label));
			test_labels.push_back((double)label);
		}
		printf("read %d test labels success.\n", test_labels.size());
	}
}

void data_strengthen()
{
	int i = 0;
	int num = 0;
	srand((int)time(0));
	for (i = 0; i < 60000; i++)
	{
		//        if((rand() % (100) / 100.0)<0.3)
		//            {
		//            RotationLeft90(train_images[i]);
		//            strengthen_labels.push_back(train_labels[i]);
		//            num++;
		//            }

		//        if((rand() % (100) / 100.0)<0.3)
		//            {
		//            RotationRight90(train_images[i]);
		//            strengthen_labels.push_back(train_labels[i]);
		//            num++;
		//            }
		//
		//        if((rand() % (100) / 100.0)<0.3)
		//            {
		//            RotationDown(train_images[i]);
		//            strengthen_labels.push_back(train_labels[i]);
		//            num++;
		//            }
		if ((rand() % (100) / 100.0) < 0.3)
		{
			geometryTrans1(train_images[i]);
			strengthen_labels.push_back(train_labels[i]);
			num++;
		}
		if ((rand() % (100) / 100.0) < 0.3)
		{
			geometryTrans2(train_images[i]);
			strengthen_labels.push_back(train_labels[i]);
			num++;
		}
	}
	train_images.insert(train_images.end(), strengthen_images.begin(), strengthen_images.end());
	train_labels.insert(train_labels.end(), strengthen_labels.begin(), strengthen_labels.end());

	printf("%d\n", num);
}

// -------------------------------------神经网络函数定义----------------------------------

// 为权重矩阵和偏置向量随机赋初值
void init_parameters() {
	for (int i = 0; i < 784; i++) {
		for (int j = 0; j < nh; j++) w1[i][j] = rand() / (10 * (double)RAND_MAX) - 0.05;
	}

	for (int i = 0; i < nh; i++) {
		for (int j = 0; j < 10; j++) w2[i][j] = rand() / (10 * (double)RAND_MAX) - 0.05;
	}

	for (int i = 0; i < nh; i++) bias1[i] = rand() / (10 * (double)RAND_MAX) - 0.1;
	for (int i = 0; i < 10; i++) bias2[i] = rand() / (10 * (double)RAND_MAX) - 0.1;
}
// 激活函数 sigmoid，可以把 x 映射到 0 ~ 1 之间
// s(x) = 1 / (1 + e^(-x))
// s'(x) = s(x) * (1 - s(x))
double sigmoid(double x) {
	return 1 / (1 + exp(-x));
}

// 激活函数 Leaky_Relu
double Leaky_Relu(double x) {
	return (x > 0 ? x : 0.01 * x);
}

// 激活函数 Relu
double Relu(double x) {
	return (x > 0 ? x : 0);
}

// 通过图像的输入得到隐藏层的输出
vector<double> get_hidden_out(vector<double>& image) {
	vector<double> hidden_out(nh);
	for (int i = 0; i < nh; i++) {          // 对于每一个隐藏单元
		double sum = 0.0;
		for (int j = 0; j < 784; j++) {
			sum += image[j] * w1[j][i];
		}
		hidden_out[i] = sigmoid(sum + bias1[i]);
	}
	return hidden_out;
}
// 通过隐藏层的输出得到网络最终的输出
vector<double> get_z(vector<double>& hidden_out) {
	vector<double> z(10);
	for (int i = 0; i < 10; i++) {        // 对于每一个输出单元
		double sum = 0.0;
		for (int j = 0; j < nh; j++) {
			sum += hidden_out[j] * w2[j][i];
		}
		z[i] = sigmoid(sum + bias2[i]);
	}
	return z;
}
// 计算损失函数（1/2 均方误差）
double get_loss(vector<double>& z, double label) {
	double loss = 0;
	int true_label = (int)label;
	for (int i = 0; i < 10; i++) {
		if (i != true_label) loss += z[i] * z[i];
		else loss += (1 - z[i]) * (1 - z[i]);
	}
	return loss / 2;
}
void train_and_test() {
	for (e = 1; e <= epoch; e++) {
		for (int im = 0; im < train_images.size(); im++) {
			double grad_w1[784][nh];
			double grad_w2[nh][10];
			double grad_bias1[nh];
			double grad_bias2[10];

			vector<double> hidden_out = get_hidden_out(train_images[im]);
			vector<double> z = get_z(hidden_out);
			double loss = get_loss(z, train_labels[im]);
			int true_label = (int)train_labels[im];
			//if (im % 1000 == 0) printf("epoch = %d, image = %d, loss = %f\n", e, im + 1, loss);

			// -------------------------------------计算梯度----------------------------------

			for (int j = 0; j < nh; j++) {
				double sum = 0.0;
				for (int k = 0; k < 10; k++) {
					double labelk = (k == true_label) ? 1.0 : 0.0;
					sum += (z[k] - labelk) * z[k] * (1 - z[k]) * w2[j][k] * hidden_out[j] * (1 - hidden_out[j]);
				}
				grad_bias1[j] = sum;
			}

			for (int i = 0; i < 784; i++) {
				for (int j = 0; j < nh; j++) {
					double sum = 0.0;
					grad_w1[i][j] = grad_bias1[j] * train_images[im][i];
				}
			}

			for (int k = 0; k < 10; k++) {
				double labelk = (k == true_label) ? 1.0 : 0.0;
				grad_bias2[k] = (z[k] - labelk) * z[k] * (1 - z[k]);
			}

			for (int j = 0; j < nh; j++) {
				for (int k = 0; k < 10; k++) {
					double labelk = (k == true_label) ? 1.0 : 0.0;
					grad_w2[j][k] = grad_bias2[k] * hidden_out[j];
				}
			}

			// -------------------------------------更新权与偏置----------------------------------

			for (int i = 0; i < 784; i++) {
				for (int j = 0; j < nh; j++) {
					w1[i][j] -= learning_rate * grad_w1[i][j];
				}
			}

			for (int i = 0; i < nh; i++) {
				for (int j = 0; j < 10; j++) {
					w2[i][j] -= learning_rate * grad_w2[i][j];
				}
			}

			for (int i = 0; i < nh; i++) {
				bias1[i] -= learning_rate * grad_bias1[i];
			}

			for (int i = 0; i < 10; i++) {
				bias2[i] -= learning_rate * grad_bias2[i];
			}
		}
		test();
	}
}
void test() {
	int cnt = 0, flag = 0;
	printf("  Press 'ENTER' to check at every single picture\nor 'space' to skip right ones \nor 'ESC' to look at the precision rate immediately.\n");
	for (int i = 0; i < test_images.size(); i++) {
		int true_label = (int)test_labels[i];
		vector<double> hidden_out = get_hidden_out(test_images[i]);
		vector<double> z = get_z(hidden_out);
		int recognize = -1;
		double max = 0;
		for (int i = 0; i < 10; i++) {
			if (z[i] > max) {
				max = z[i];
				recognize = i;
			}
		}
		if (recognize == true_label)
		{
			int ch = 0;
			if (flag < 1)
			{
				show(i);
				printf("right one,label is %d\n", true_label);
				ch = _getch();
				if (ch == 32)
					flag = 1;
				else if (ch == 27)
					flag = 2;
			}
			cnt++;
		}
		else
		{
			int ch = 0;
			if (flag < 2)
			{
				show(i);
				printf("wrong one,label is %d,but the recognize result is %d\n", true_label, recognize);
				ch = _getch();
				if (ch == 27)
					flag = 2;
			}
		}
		progress_bar(640, 480, 920, 480 + 20, ((double)i / (double)(test_images.size())));
	}
	printf("the precision of neural network = %f\n", (double)cnt / test_images.size());
}

void show(int index)
{
	DWORD* pMem = GetImageBuffer();
	int x, y, tx, ty, t = 20, sum;// 循环变量
	for (y = 0; y < 28; y += 1)// 处理每一个小方块
	{
		for (x = 0; x < 28; x += 1)
		{
			sum = test_images[index][28 * y + x] * 256;
			for (ty = 0; ty < t; ty++)
			{
				for (tx = 0; tx < t; tx++)
				{
					pMem[((y + 1) * 20 + ty) * 966 + (x + 1) * 20 + tx] = RGB(sum, sum, sum);
				}
			}
		}
	}
}
void write_parameters()
{
	ofstream fileout;
	fileout.open("Neural-Network-parameter.txt", ios::out);
	if (!fileout.is_open())
		cout << "Open file failure" << endl;
	for (int i = 0; i < 784; i++)
	{
		for (int j = 0; j < nh; j++)
		{
			fileout << w1[i][j] << " ";
		}
	}
	for (int i = 0; i < nh; i++)
	{
		for (int j = 0; j < 10; j++)
		{
			fileout << w2[i][j] << " ";
		}
	}
	for (int i = 0; i < nh; i++)
	{
		fileout << bias1[i] << " ";
	}
	for (int i = 0; i < 10; i++)
	{
		fileout << bias2[i] << " ";
	}
	fileout.close();
}
void read_parameters()
{
	ifstream filein;
	filein.open("Neural-Network-parameter.txt", ios::in);
	if (!filein.is_open())
		cout << "Open file failure" << endl;
	for (int i = 0; i < 784; i++)
	{
		for (int j = 0; j < nh; j++)
		{
			filein >> w1[i][j];
		}
	}
	for (int i = 0; i < nh; i++)
	{
		for (int j = 0; j < 10; j++)
		{
			filein >> w2[i][j];
		}
	}
	for (int i = 0; i < nh; i++)
	{
		filein >> bias1[i];
	}
	for (int i = 0; i < 10; i++)
	{
		filein >> bias2[i];
	}
	filein.close();
	printf("read parameter of neural network success.\n");
}
void train_and_save_parameters()
{
	read_train_images();                // 载入训练集图像
	read_train_labels();                // 载入训练集标签
	read_test_images();                 // 载入测试集图像
	read_test_labels();                 // 载入测试集标签
	init_parameters();                  // 为权重矩阵和偏置向量随机赋初值
	train_and_test();
	write_parameters();
}

int recognize(double(*p)[28])
{
	vector<double>number_data(28);
	for (int i = 0; i < 28; i++)
	{
		for (int j = 0; j < 28; j++)
		{
			number_data.push_back(p[i][j]);
		}
	}

	vector<double> hidden_out = get_hidden_out(number_data);
	vector<double> z = get_z(hidden_out);
	int recognize = -1;
	double max = 0;
	for (int i = 0; i < 10; i++) {
		if (z[i] > max) {
			max = z[i];
			recognize = i;
		}
	}
	return recognize;
}

//-------------------------------------卷积神经网络函数定义----------------------------------

void init_parameters2() {
	for (int layer = 0; layer < layer_number; layer++) {
		for (int i = 0; i < convolution_size; i++) {
			for (int j = 0; j < convolution_size; j++) {
				weight1[layer][i][j] = rand() / (10 * (double)RAND_MAX) - 0.05;
			}
		}
	}
	for (int layer = 0; layer < layer_number; layer++) {
		layer_bias[layer] = rand() / (10 * (double)RAND_MAX) - 0.05;
	}
	for (int k = 0; k < 10; k++) {
		for (int layer = 0; layer < layer_number; layer++) {
			for (int i = 0; i < size2; i++) {
				for (int j = 0; j < size2; j++) {
					weight2[k][layer][i][j] = rand() / (10 * (double)RAND_MAX) - 0.05;
				}
			}
		}
	}
	for (int k = 0; k < 10; k++) {
		pool_bias[k] = rand() / (10 * (double)RAND_MAX) - 0.05;
	}
}
vector<vector<double>>get_convolution_out(vector<double>& image)
{
	vector<double>convolution_everyLayer(size1 * size1);
	vector<vector<double>>convolution_out;
	for (int layer = 0; layer < layer_number; layer++)
	{
		for (int k = 0; k < size1 * size1; k++)
		{
			int x = k / size1, y = k % size1;
			double sum = 0.0;
			for (int i = 0; i < convolution_size; i++)
			{
				for (int j = 0; j < convolution_size; j++)
				{
					sum += image[28 * (x + i) + y + j] * weight1[layer][i][j];
				}
			}
			convolution_everyLayer[k] = sigmoid(sum + layer_bias[layer]);
		}
		convolution_out.push_back(convolution_everyLayer);
	}
	return convolution_out;
}
vector<vector<int>>get_maxposition(vector<vector<double>>& convolution_out)
{
	vector<vector<int>>maxPosition;
	vector<int> max_onelayer(size2 * size2);
	double max = 0, thisOne = 0;
	for (int layer = 0; layer < layer_number; layer++)
	{
		for (int i = 0; i < size2; i++)
		{
			for (int j = 0; j < size2; j++)
			{
				max = convolution_out[layer][(i * size1 + j) * press_rate];
				for (int k = 1; k < press_rate * press_rate; k++)
				{
					int x = k / press_rate, y = k % press_rate;
					thisOne = convolution_out[layer][size1 * (i * press_rate + x) + press_rate * j + y];
					if (thisOne > max)
					{
						max = thisOne;
						max_onelayer[size2 * i + j] = k;
					}
				}
			}
		}
		maxPosition.push_back(max_onelayer);
	}
	return maxPosition;
}
vector<vector<double>>get_pooling_out(vector<vector<int>>& maxPosition,
	vector<vector<double>>& convolution_out)
{
	vector<double>every_pooling(size2 * size2);
	vector<vector<double>>pooling_out;
	int k = 0, x = 0, y = 0;
	for (int layer = 0; layer < layer_number; layer++)
	{
		for (int i = 0; i < size2; i++)
		{
			for (int j = 0; j < size2; j++)
			{
				k = maxPosition[layer][i * size2 + j];
				x = k / press_rate;
				y = k % press_rate;
				every_pooling[size2 * i + j] =
					convolution_out[layer][size1 * (i * press_rate + x) + press_rate * j + y];
			}
		}
		pooling_out.push_back(every_pooling);
	}
	return pooling_out;
}
vector<double>get_result(vector<vector<double>>& pooling_out)
{
	vector<double> result(10);
	for (int k = 0; k < 10; k++)
	{
		double sum = 0.0;
		for (int layer = 0; layer < layer_number; layer++)
		{
			for (int i = 0; i < size2; i++)
			{
				for (int j = 0; j < size2; j++)
				{
					sum += pooling_out[layer][i * size2 + j] * weight2[k][layer][i][j];
				}
			}
		}
		result[k] = sigmoid(sum + pool_bias[k]);
	}
	return result;
}
void train_and_test2()
{
	for (e = 1; e <= epoch2; e++)
	{
		for (int im = 0; im < train_images.size(); im++)
		{
			double grad_weight1[layer_number][convolution_size][convolution_size] = { 0 };
			double grad_layer_bias[layer_number] = { 0 };
			double grad_weight2[10][layer_number][size2][size2] = { 0 };
			double grad_pool_bias[10] = { 0 };

			vector<vector<double>>convolution_out = get_convolution_out(train_images[im]);
			vector<vector<int>>maxposition = get_maxposition(convolution_out);
			vector<vector<double>>pooling_out = get_pooling_out(maxposition, convolution_out);
			vector<double>result = get_result(pooling_out);
			double loss = get_loss(result, train_labels[im]);
			int true_label = (int)train_labels[im];

			//------------------------------------ - 计算梯度----------------------------------

			vector<double>unit_error_oneLayer(size1 * size1);
			vector<vector<double>>neural_unit_error;

			for (int k = 0; k < 10; k++)
			{
				double labelk = (k == true_label) ? 1.0 : 0.0;
				grad_pool_bias[k] = (result[k] - labelk) * result[k] * (1 - result[k]);
			}

			for (int k = 0; k < 10; k++)
			{
				for (int layer = 0; layer < layer_number; layer++)
				{
					for (int i = 0; i < size2; i++)
					{
						for (int j = 0; j < size2; j++)
						{
							grad_weight2[k][layer][i][j] = grad_pool_bias[k] * convolution_out[layer][i * size2 + j];
						}
					}
				}
			}

			for (int layer = 0; layer < layer_number; layer++)
			{
				double unit_error_sum = 0.0;
				for (int i = 0; i < size2; i++)
				{
					for (int j = 0; j < size2; j++)
					{
						double sum = 0.0;
						for (int k = 0; k < 10; k++)
						{
							sum += weight2[k][layer][i][j] * grad_pool_bias[k];
						}
						int position = maxposition[layer][size2 * i + j];
						int x = position / press_rate, y = position % press_rate;
						int index = size1 * (i * press_rate + x) + press_rate * j + y;

						unit_error_oneLayer[index] = sum * convolution_out[layer][index] * (1 - convolution_out[layer][index]);

						unit_error_sum += unit_error_oneLayer[index];
					}
				}
				neural_unit_error.push_back(unit_error_oneLayer);

				grad_layer_bias[layer] = unit_error_sum;
			}

			for (int layer = 0; layer < layer_number; layer++)
			{
				for (int i = 0; i < convolution_size; i++)
				{
					for (int j = 0; j < convolution_size; j++)
					{
						double sum = 0.0;
						for (int x = 0; x < size1; x++)
						{
							for (int y = 0; y < size1; y++)
							{
								sum += neural_unit_error[layer][x * size1 + y] * train_images[im][(i + x) * 28 + j + y];
							}
						}
						grad_weight1[layer][i][j] = sum;
					}
				}
			}

			// -------------------------------------更新权与偏置----------------------------------

			for (int layer = 0; layer < layer_number; layer++)
			{
				for (int i = 0; i < convolution_size; i++)
				{
					for (int j = 0; j < convolution_size; j++)
					{
						weight1[layer][i][j] -= learning_speed * grad_weight1[layer][i][j];
					}
				}
				for (int i = 0; i < size1; i++)
				{
					for (int j = 0; j < size1; j++)
					{
						layer_bias[layer] -= learning_speed * grad_layer_bias[layer];
					}
				}
			}

			for (int k = 0; k < 10; k++)
			{
				for (int layer = 0; layer < layer_number; layer++)
				{
					for (int i = 0; i < size2; i++)
					{
						for (int j = 0; j < size2; j++)
						{
							weight2[k][layer][i][j] -= learning_speed * grad_weight2[k][layer][i][j];
						}
					}
				}
			}

			for (int k = 0; k < 10; k++)
			{
				pool_bias[k] -= learning_speed * grad_pool_bias[k];
			}

		}
		test2();
	}
}
void test2() {
	int cnt = 0, flag = 0;
	for (int i = 0; i < test_images.size(); i++) {
		vector<vector<double>>convolution_out = get_convolution_out(test_images[i]);
		vector<vector<int>>maxposition = get_maxposition(convolution_out);
		vector<vector<double>>pooling_out = get_pooling_out(maxposition, convolution_out);
		vector<double>result = get_result(pooling_out);
		int true_label = (int)test_labels[i];

		int recognize = -1;
		double max = 0;
		for (int i = 0; i < 10; i++) {
			if (result[i] > max) {
				max = result[i];
				recognize = i;
			}
		}
		if (recognize == true_label)
		{
			int ch = 0;
			if (flag < 1)
			{
				show(i);
				printf("right one,label is %d\n", true_label);
				ch = _getch();
				if (ch == 32)
					flag = 1;
				else if (ch == 27)
					flag = 2;
			}
			cnt++;
		}
		else
		{
			int ch = 0;
			if (flag < 2)
			{
				show(i);
				printf("wrong one,label is %d,but the recognize result is %d\n", true_label, recognize);
				ch = _getch();
				if (ch == 27)
					flag = 2;
			}
		}
		progress_bar(640, 480, 920, 480 + 20, ((double)i / (double)(test_images.size())));
	}
	//printf("epoch = %d, precision = %f\n", e, (double)cnt / test_images.size());
	printf("the precision of convolution  neural network = %f\n", (double)cnt / test_images.size());
}

void write_parameters2()
{
	ofstream fileout;
	fileout.open("c", ios::out);
	if (!fileout.is_open())
		cout << "Open file failure" << endl;
	for (int i = 0; i < layer_number; i++)
	{
		for (int j = 0; j < convolution_size; j++)
		{
			for (int k = 0; k < convolution_size; k++)
			{
				fileout << weight1[i][j] << " ";
			}

		}
	}
	for (int i = 0; i < 10; i++)
	{
		for (int j = 0; j < layer_number; j++)
		{
			for (int k = 0; k < size2; k++)
			{
				for (int m = 0; m < size2; m++)
				{
					fileout << weight2[i][j] << " ";
				}
			}
		}
	}
	for (int i = 0; i < layer_number; i++)
	{
		fileout << layer_bias[i] << " ";
	}
	for (int i = 0; i < 10; i++)
	{
		fileout << pool_bias[i] << " ";
	}
	fileout.close();
}
void read_parameters2()
{
	ifstream filein;
	filein.open("Convolution-Network-parameter.txt", ios::in);
	if (!filein.is_open())
		cout << "Open file failure" << endl;
	for (int i = 0; i < layer_number; i++)
	{
		for (int j = 0; j < convolution_size; j++)
		{
			for (int k = 0; k < convolution_size; k++)
			{
				filein >> weight1[i][j][k];
			}

		}
	}
	for (int i = 0; i < 10; i++)
	{
		for (int j = 0; j < layer_number; j++)
		{
			for (int k = 0; k < size2; k++)
			{
				for (int m = 0; m < size2; m++)
				{
					filein >> weight2[i][j][k][m];
				}
			}
		}
	}
	for (int i = 0; i < layer_number; i++)
	{
		filein >> layer_bias[i];
	}
	for (int i = 0; i < 10; i++)
	{
		filein >> pool_bias[i];
	}
	filein.close();
	printf("read parameter of convolution neural network success.\n");
}
void train_and_save_parameters2()
{
	init_parameters2();                  // 为权重矩阵和偏置向量随机赋初值
	train_and_test2();
	write_parameters2();
}

int recognize2(double(*p)[28])
{
	vector<double>number_data(28);
	for (int i = 0; i < 28; i++)
	{
		for (int j = 0; j < 28; j++)
		{
			number_data.push_back(p[i][j]);
		}
	}

	vector<vector<double>>convolution_out = get_convolution_out(number_data);
	vector<vector<int>>maxposition = get_maxposition(convolution_out);
	vector<vector<double>>pooling_out = get_pooling_out(maxposition, convolution_out);
	vector<double>result = get_result(pooling_out);
	int recognize = -1;
	double max = 0;
	for (int i = 0; i < 10; i++) {
		if (result[i] > max) {
			max = result[i];
			recognize = i;
		}
	}
	return recognize;
}

//-----------------------------------------------------------------------
int main() {
	printf("‘识别1’与‘识别2’分别指使用神经网络和卷积神经网络进行识别，\n");
	printf("识别前须先进行‘处理’将图片进行马赛克处理并存入相应数组，清除可刷新整个界面。\n");
	printf("‘测试1’与‘测试2’分别指使用神经网络和卷积神经网络对测试数据进行测试，\n");
	printf("测试过程可使用'enter'逐一查看测试图像，按'space'可跳过识别正确项，按'esc'跳过查看图像过程\n");
	//train_and_save_parameters();
	//train_and_save_parameters2();
	read_parameters();
	read_parameters2();
	read_test_images();                 // 载入测试集图像
	read_test_labels();                 // 载入测试集标签
//	data_strengthen();               //数据增强
	initgraph(966, 600, SHOWCONSOLE);
	fun();
	//system("pause");
	return 0;
}

void fun()
{
	MOUSEMSG m;
	int c, i, j, k, t, z1 = 0, n = 0, num;
	long int x[150] = { 0 }, y[150] = { 0 };//写数字的时候隔一定距离就采一次样，隔20存1次点的坐标，画布本身只有20*28的边长
	double number[28][28] = { 0.0 };
	DWORD* pMem = GetImageBuffer();//获取默认的绘图窗口的显存指针
	while (1)
	{
		BeginBatchDraw();
		setwritemode(R2_COPYPEN);
		for (i = 0; i < 600; i++)
		{
			setlinecolor(HSLtoRGB(120 + (float)(i) / 600.0 * 120.0, 1, 0.5));
			line(0, i, 966, i);//从上往下划线，颜色由绿变蓝
		}
		button_1(600 + 40, 30, 600 + 40 + 120, 90);
		button_1(600 + 40 * 2 + 120, 30, 600 + 40 * 2 + 120 * 2, 90);
		button_1(600 + 40, 120, 600 + 40 + 120, 180);
		button_1(600 + 40 * 2 + 120, 120, 600 + 40 * 2 + 120 * 2, 180);
		button_1(600 + 40, 210, 600 + 40 + 120, 270);
		button_1(600 + 40 * 2 + 120, 210, 600 + 40 * 2 + 120 * 2, 270);
		setbkmode(TRANSPARENT); //透明背景模式显示文字
		settextstyle(40, 0, _T("楷体"));
		settextcolor(BLACK);
		outtextxy(640 + 13, 30 + 10, _T("识别1"));
		outtextxy(800 + 13, 30 + 10, _T("识别2"));
		outtextxy(640 + 20, 120 + 10, _T("清除"));
		outtextxy(800 + 20, 120 + 10, _T("处理"));
		outtextxy(640 + 13, 210 + 10, _T("测试1"));
		outtextxy(800 + 13, 210 + 10, _T("测试2"));
		setfillcolor(WHITE);
		solidrectangle(0, 0, 605, 600);
		setfillcolor(BLACK);
		solidrectangle(20, 20, 580, 580);
		EndBatchDraw();

		n = 0;
		z1 = 0;
		while (1)//循环直到鼠标点到按键
		{
			setfillcolor(WHITE);
			m = GetMouseMsg();//和下一个函数结合可实现按下鼠标左键连续画线
			c = (GetAsyncKeyState(VK_LBUTTON) & 0x8000);//非阻塞函数，随时记录鼠标按键状态
			if (c != 0)
			{
				if (m.x > 20 && m.y > 20 && m.x < 580 && m.y < 580)//画板部分
				{
					if (z1 == 0)
					{
						x[0] = m.x;
						y[0] = m.y;
						z1++;//第一个点只画一次
					}
					if ((m.x - x[n]) * (m.x - x[n]) + (m.y - y[n]) * (m.y - y[n]) > 400 && n < 150)//距离大于20的时候记录坐标
					{//这样手写的数字的信息就可以用一些坐标表示了
						x[n + 1] = m.x;
						y[n + 1] = m.y;
						n++;//n表示已存起来的点的数量
					}
					solidcircle(m.x, m.y, 15);
				}
				else if (m.x > 600 + 40 * 2 + 120 && m.y > 120 && m.x < 600 + 40 * 2 + 120 * 2 && m.y < 180)//处理
				{
					current_button(3);
					for (t = 0; t < n - 1; t++)
					{
						//setwritemode(R2_MERGEPEN);
						int distance = (x[t + 1] - x[t]) * (x[t + 1] - x[t]) + (y[t + 1] - y[t]) * (y[t + 1] - y[t]);
						if (distance < 14400)
						{
							double d = (double)distance;
							setlinecolor(RGB(255, 255, 255));
							setlinestyle(PS_JOIN_ROUND | PS_ENDCAP_ROUND, 70 - 50 * sigmoid((d - 480.0) / 300.0));//笔画越快，线条越细
							//printf("%.2lf ",(d - 450.0) / 300.0);
							line(x[t], y[t], x[t + 1], y[t + 1]);
							//用于查看所取的点
							//setfillcolor(RED);
							//solidcircle(x[t], y[t], 5);
						}
					}
					mosaic(pMem, 20, number);//马赛克处理的步骤，只是用于获取各小格的平均值，不用画到屏幕上
				}
				else if (m.x > 600 + 40 && m.y > 30 && m.x < 600 + 40 + 120 && m.y < 90)//神经网络识别数字
				{
					current_button(0);
					int result;
					setfillcolor(WHITE);
					solidrectangle(640, 300, 890, 440);
					settextstyle(40, 0, _T("楷体"));
					settextcolor(BLUE);
					outtextxy(640 + 10, 300 + 5, _T("这数是:"));
					result = recognize(number);
					TCHAR a[10];
					_stprintf_s(a, _T("%d"), result);
					settextstyle(80, 0, _T("宋体"));
					settextcolor(RED);
					outtextxy(680 + 60, 320 + 30, a);
				}
				else if (m.x > 600 + 40 * 2 + 120 && m.y > 30 && m.x < 600 + 40 * 2 + 120 * 2 && m.y < 90)//卷积神经网络识别数字
				{
					current_button(1);
					int result;
					setfillcolor(WHITE);
					solidrectangle(640, 300, 890, 440);
					settextstyle(40, 0, _T("楷体"));
					settextcolor(BLUE);
					outtextxy(640 + 10, 300 + 5, _T("这数是:"));
					result = recognize2(number);
					TCHAR a[10];
					_stprintf_s(a, _T("%d"), result);
					settextstyle(80, 0, _T("宋体"));
					settextcolor(RED);
					outtextxy(680 + 60, 320 + 30, a);
				}
				else if (m.x > 600 + 40 && m.y > 210 && m.x < 600 + 40 + 120 && m.y < 270)//测试
				{
					current_button(4);
					test();
				}
				else if (m.x > 600 + 40 * 2 + 120 && m.y > 210 && m.x < 600 + 40 * 2 + 120 * 2 && m.y < 270)
				{
					current_button(5);
					test2();
				}
				else if (m.x > 600 + 40 && m.y > 120 && m.x < 600 + 40 + 120 && m.y < 180)//清除
				{
					current_button(2);
					break;//清空，即返回大的循环，重新书写数字
				}
			}
		}
	}
}
void button_1(int x1, int y1, int x2, int y2)//画出按钮
{
	setfillcolor(RGB(128, 128, 128));
	solidrectangle(x1, y1, x2, y2);
	setlinestyle(PS_SOLID | PS_JOIN_BEVEL | PS_ENDCAP_SQUARE, 5);//线形为实线，连接处为斜面，端点为方形，线宽5，作出按钮边框
	setlinecolor(RGB(192, 192, 192));//浅灰色线
	line(x1, y1, x1, y2);//左边线
	line(x1, y1, x2, y1);//上方边线
	setlinecolor(RGB(64, 64, 64));//深灰色线，相交部分覆盖浅灰色线
	line(x2, y1, x2, y2);//右边线
	line(x1, y2, x2, y2);//下方线
	setfillcolor(RGB(192, 192, 192));//用浅灰色三角形处理，增加按钮立体感
	POINT pts_1[] = { {x1 - 2, y2 + 2}, {x1 + 2, y2 - 2}, {x1 - 2, y1 - 2} };
	solidpolygon(pts_1, 3);//左下角
	POINT pts_2[] = { {x2 - 2, y1 + 2}, {x2 + 2, y1 - 2}, {x2 - 2, y1 - 2} };
	solidpolygon(pts_2, 3);//右上角
}
void mosaic(DWORD* pMem, int t, double(*p)[28])
{
	int sum;// 颜色值的和
	int  color;// 每个小方块内的像素数量,每个像素的颜色
	int x, y, tx, ty;// 循环变量
	for (y = 0; y < 28; y += 1)// 处理每一个小方块
	{
		for (x = 0; x < 28; x += 1)
		{
			sum = 0;
			for (ty = 0; ty < t; ty++)
			{
				for (tx = 0; tx < t; tx++)
				{
					color = pMem[((y + 1) * 20 + ty) * 966 + (x + 1) * 20 + tx];
					sum += GetRValue(color);
				}
			}
			sum /= t * t;// 求红、绿、蓝颜色的平均值
			p[y][x] = (double)sum / 256.0;
			for (ty = 0; ty < t; ty++)//将马赛克处理后的效果在屏幕上显示
			{
				for (tx = 0; tx < t; tx++)
				{
					pMem[((y + 1) * 20 + ty) * 966 + (x + 1) * 20 + tx] = RGB(sum, sum, sum);
				}
			}
		}
	}
}
void progress_bar(int x1, int y1, int x2, int y2, double progress)
{
	setfillcolor(GREEN);
	setlinecolor(YELLOW);
	setlinestyle(PS_SOLID, 3);
	rectangle(x1 - 2, y1 - 2, x2 + 1, y2 + 2);
	solidrectangle(x1, y1, x1 + (int)(progress * ((double)(x2 - x1))), y2);
}
void current_button(int function)
{
	//setwritemode(R2_COPYPEN);
	int side = 6, x0 = 640, y0 = 30, dx = 160, dy = 90, wide = 120, high = 60;
	int x, y;
	setlinestyle(PS_SOLID, 3);
	setlinecolor(RGB(128, 128, 128));
	for (int i = 0; i < 2; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			rectangle(x0 + side + i * dx, y0 + side + j * dy, x0 + wide - side + i * dx, y0 + high - side + j * dy);
		}
	}
	x = function % 2, y = function / 3;
	setlinecolor(YELLOW);
	rectangle(x0 + side + x * dx, y0 + side + y * dy, x0 + wide - side + x * dx, y0 + high - side + y * dy);
}