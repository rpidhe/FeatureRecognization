#include<iostream>
#include<fstream>
#include<string>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <io.h>
#include <direct.h>
using namespace std;
using namespace cv;
void rgb2luv(int *RGB, float *LUV, int size)
{
	for (int i = 0; i < size; i++)
	{
		LUV[i] = RGB[i];
	}
	//return;
	int i;
	double x, y, X, Y, Z, den, u2, v2, X0, Z0, Y0, u20, v20, r, g, b;

	X0 = (0.607 + 0.174 + 0.201);
	Y0 = (0.299 + 0.587 + 0.114);
	Z0 = (0.066 + 1.117);

	/* Y0 = 1.0 */
	u20 = 4 * X0 / (X0 + 15 * Y0 + 3 * Z0);
	v20 = 9 * Y0 / (X0 + 15 * Y0 + 3 * Z0);

	for (i = 0; i<size; i += 3)
	{
		if (RGB[i] <= 20)  r = (double)(8.715e-4*RGB[i]);
		else r = (double)pow((RGB[i] + 25.245) / 280.245, 2.22);

		if (RGB[i + 1] <= 20)  g = (double)(8.715e-4*RGB[i + 1]);
		else g = (double)pow((RGB[i + 1] + 25.245) / 280.245, 2.22);

		if (RGB[i + 2] <= 20)  b = (double)(8.715e-4*RGB[i + 2]);
		else b = (double)pow((RGB[i + 2] + 25.245) / 280.245, 2.22);

		X = 0.412453*r + 0.357580*g + 0.180423*b;
		Y = 0.212671*r + 0.715160*g + 0.072169*b;
		Z = 0.019334*r + 0.119193*g + 0.950227*b;

		if (X == 0.0 && Y == 0.0 && Z == 0.0)
		{
			x = 1.0 / 3.0; y = 1.0 / 3.0;
		}
		else
		{
			den = X + Y + Z;
			x = X / den; y = Y / den;
		}

		den = -2 * x + 12 * y + 3;
		u2 = 4 * x / den;
		v2 = 9 * y / den;

		if (Y>0.008856) LUV[i] = (float)(116 * pow(Y, 1.0 / 3.0) - 16);
		else LUV[i] = (float)(903.3*Y);
		LUV[i + 1] = (float)(13 * LUV[i] * (u2 - u20));
		LUV[i + 2] = (float)(13 * LUV[i] * (v2 - v20));
	}
}
void luv2rgb(int *RGB, float *LUV, int size)
{
	int i, k;
	double x, y, X, Y, Z, den, u2, v2, X0, Z0, Y0, u20, v20, vec[3];

	X0 = (0.607 + 0.174 + 0.201);
	Y0 = (0.299 + 0.587 + 0.114);
	Z0 = (0.066 + 1.117);

	/* Y0 = 1.0 */
	u20 = 4 * X0 / (X0 + 15 * Y0 + 3 * Z0);
	v20 = 9 * Y0 / (X0 + 15 * Y0 + 3 * Z0);

	for (i = 0; i<size; i += 3)
	{
		if (LUV[i]>0)
		{
			if (LUV[i]<8.0) Y = ((double)LUV[i]) / 903.3;
			else Y = pow((((double)LUV[i]) + 16) / 116.0, 3.0);
			u2 = ((double)LUV[i + 1]) / 13.0 / ((double)LUV[i]) + u20;
			v2 = ((double)LUV[i + 2]) / 13.0 / ((double)LUV[i]) + v20;

			den = 6 + 3 * u2 - 8 * v2;
			if (den<0) printf("den<0\n");
			if (den == 0) printf("den==0\n");
			x = 4.5*u2 / den;
			y = 2.0*v2 / den;

			X = x / y*Y;
			Z = (1 - x - y) / y*Y;
		}
		else { X = 0.0; Y = 0.0; Z = 0.0; }

		vec[0] = (3.240479*X - 1.537150*Y - 0.498536*Z);
		vec[1] = (-0.969256*X + 1.875992*Y + 0.041556*Z);
		vec[2] = (0.055648*X - 0.204043*Y + 1.057311*Z);
		for (k = 0; k<3; k++)
		{
			if (vec[k] <= 0.018) vec[k] = 255 * 4.5*vec[k];
			else vec[k] = 255 * (1.099*pow(vec[k], 0.45) - 0.099);
			if (vec[k]>255) vec[k] = 255;
			else if (vec[k]<0) vec[k] = 0;
			RGB[i + k] = lroundf((float)vec[k]);
		}
	}
}
double distance(Vec3f&luv1, Vec3f& luv2)
{
	return (luv1[0] - luv2[0]) * (luv1[0] - luv2[0])
		+ (luv1[1] - luv2[1]) * (luv1[1] - luv2[1])
		+ (luv1[2] - luv2[2]) * (luv1[2] - luv2[2]);
}
double calinski_harabaz_score(Mat& cluster_data, Mat& center,Mat& ImgLabel,int k_cluster)
{
	Vec3f globe_mean(0.);
	int n_data = cluster_data.rows;
	for (int i = 0; i < n_data; i++)
	{
		Vec3f p = cluster_data.at<Vec3f>(i, 0);
		globe_mean[0] += p[0];
		globe_mean[1] += p[1];
		globe_mean[2] += p[2];
	}
	globe_mean[0] /= n_data;
	globe_mean[1] /= n_data;
	globe_mean[2] /= n_data;
	vector<int> cluster_count(k_cluster,0);
	int* labelData = (int*)ImgLabel.data;
	for (int i = 0; i < n_data; i++)
	{
		cluster_count[labelData[i]] ++;
	}
	double ssb = 0;
	for (int i = 0; i < k_cluster; i++)
	{
		ssb += cluster_count[i] * distance(center.at<Vec3f>(i, 0), globe_mean);
	}
	double ssw = 0;
	for (int i = 0; i < n_data; i++) {
		ssw += distance(cluster_data.at<Vec3f>(i, 0), center.at<Vec3f>(labelData[i], 0));
	}
	double vrc = ssb / ssw *(n_data - k_cluster) / (k_cluster - 1);
	return vrc;
}
int main(int argc, char* argv[])
{
	if (argc != 4)
	{
		printf("Usage : DCDCluster.exe <k_cluster> <single_dcd_file_path> <output_dir>\n");
		return -1; 
	}
	int k_cluster = atoi(argv[1]);
	string single_dcd_file_path = argv[2];
	string save_dir = argv[3];
	if (_access(save_dir.c_str(), 0) != 0)
		_mkdir(save_dir.c_str());
	ifstream single_dcd_file(single_dcd_file_path);
	int rgb[3];
	vector<string> img_names;
	string img_name;
	vector<Vec3f> tmpData;
	while (single_dcd_file>> img_name>>rgb[0]>>rgb[1]>>rgb[2])
	{
		float luv[3];
		img_names.push_back(img_name);
		rgb2luv(rgb, luv, 3);
		tmpData.push_back(Vec3f(luv));
	}
	single_dcd_file.close();
	Mat ImgData(tmpData.size(), 1, CV_32FC3);
	Mat_<Vec3f>::iterator itData = ImgData.begin<Vec3f>();
	//将源图像的数据输入给新建的ImgData
	for (int i = 0; i < tmpData.size(); i++)
	{
		*itData = tmpData[i];
		itData++;
	}
	double max_score = 0;
	int best_k;
	if (k_cluster == -1) {
		for (int k = 3; k < 50; k++) {
			Mat ImgLabel, ImgCenter;
			kmeans(ImgData, k, ImgLabel, TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 1000, 0.01), 1, KMEANS_PP_CENTERS, ImgCenter);
			double score = calinski_harabaz_score(ImgData, ImgCenter, ImgLabel, k);
			if (max_score < score)
			{
				max_score = score;
				best_k = k;
			}
			cout << "K=" << k << " score=" << score << endl;
		}
	}
	else {
		best_k = k_cluster;
	}
	cout << "Best K=" << best_k << endl;
	Mat ImgLabel, ImgCenter;
	kmeans(ImgData, best_k, ImgLabel, TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 1000, 0.01), 1, KMEANS_PP_CENTERS, ImgCenter);
	vector<vector<string> > clusters(best_k);
	int* labelData = (int*)ImgLabel.data;
	for (int i = 0; i < img_names.size(); i++)
	{
		clusters[labelData[i]].push_back(img_names[i]);
	}
	
	/*for (int i = 0; i < best_k; i++)
	{
		char strC[5];
		_itoa(i, strC, 10);
		string cluster_img_save_dir = save_dir + strC + "/";
		if (_access(cluster_img_save_dir.c_str(), 0) != 0)
			_mkdir(cluster_img_save_dir.c_str());
		for (string img_name : clusters[i])
		{
			Mat img = imread(img_dir + img_name);
			imwrite(cluster_img_save_dir + img_name, img);
		}
	}*/
	for (int i = 0; i < best_k; i++)
	{
		char strC[5];
		_itoa(i, strC, 10);
		ofstream cluster_file(save_dir + strC + ".txt", ios::out);
		for (string img_name : clusters[i])
		{
			cluster_file << img_name.substr(0,img_name.length()-4) << endl;
		}
		cluster_file.close();
	}
	ofstream center_file(save_dir + "centers.txt", ios::out);

	Mat ImgCenterRgb(ImgCenter.size(), CV_32SC3);
	//luv2rgb((int*)ImgCenterRgb.data, (float*)ImgCenter.data, ImgCenter.rows * 3);
	float* centerData = (float*)ImgCenter.data;
	for (int i = 0; i < best_k; i++)
	{
		//Vec3i rgb = ImgCenterRgb.at<Vec3i>(i, 0);
		int rbg[3];
		float luv[3] = { centerData[3 * i], centerData[3 * i + 1], centerData[3 * i + 2] };
		luv2rgb(rgb, luv, 3);
		center_file << rgb[0] <<' '<< rgb[1] << ' ' << rgb[2] << endl;
	}
	center_file.close();
}