#include<iostream>
#include<fstream>
#include<string>
#include <vector>
#include <io.h>
#include <direct.h>
#include <cassert>
#include <algorithm>
#include<cmath>
#define FOR_TEST_MATCH
#define uchar unsigned char 
#define BIG_CLS 3
using namespace std;
#define RGB_COLOR_DIFF_THRESHOLD 80
#define LUV_COLOR_DIFF_THRESHOLD 200
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
struct DCDComponent
{
	int r, g, b;
	float l,u,v;
	float percent;
	bool in_luv_space;
	static bool cmp(const DCDComponent& dc1, const DCDComponent& dc2)
	{
		return dc1.percent > dc2.percent;
	}
	float distance(const DCDComponent& dc2)
	{
		if(in_luv_space)
			return float(l - dc2.l)*(l - dc2.l) +
			float(u - dc2.u)*(u - dc2.u) +
			float(v - dc2.v)*(v - dc2.v);
		return float(r - dc2.r)*(r - dc2.r) +
			float(g - dc2.g)*(g - dc2.g) +
			float(b - dc2.b)*(b - dc2.b);
	}
	void toLuv()
	{
		float luv[3];
		int rgb[3] = { r,g,b };
		rgb2luv(rgb, luv, 3);
		l = luv[0];
		u = luv[1];
		v = luv[2];
	}
};
float color_distance(DCDComponent& c1, DCDComponent& c2)
{
	assert(c1.in_luv_space == c2.in_luv_space);
	float T = c1.in_luv_space ? LUV_COLOR_DIFF_THRESHOLD : RGB_COLOR_DIFF_THRESHOLD;
	float d = c1.distance(c2);
	if (d <= T)
		return 1.0 - d * (1 / T);
	return 0.0;
}
float calculate_dcd_similarity(vector<DCDComponent>& descs1, vector<DCDComponent>&  descs2)
{
	float sim = 0;
	int dominate_color_num = descs1.size() < descs2.size()? descs1.size(): descs2.size();
	for (int i = 0; i < dominate_color_num; i++)
	{
		float p1 = descs1[i].percent;
		float p2 = descs2[i].percent;
		float S = (1 - abs(p1 - p2)) * (p1 < p2 ? p1 : p2);
		sim += color_distance(descs1[i], descs2[i]) * S;
	}
	return sim;
}

void read_file_list(std::string basepath, vector<std::string> & img_names)
{
	long long hFile = 0;
	struct _finddata_t fileInfo;
	std::string pathName;

	hFile = _findfirst(pathName.assign(basepath).append("\\*").c_str(), &fileInfo);

	if (hFile == -1) {
		cout << "Failed to find first file while read file list!\n";
		return;
	}

	do {
		std::string filename = fileInfo.name;
		if (strcmp(fileInfo.name, ".") == 0 || strcmp(fileInfo.name, "..") == 0 || (fileInfo.attrib & _A_SUBDIR))
			continue;

		std::string postfix = filename.substr(filename.length() - 3, filename.length());	//后缀
		if (postfix.compare("txt") == 0) {
			img_names.push_back(filename);
		}
	} while (_findnext(hFile, &fileInfo) == 0);

	_findclose(hFile);    // 关闭搜索句柄
}
int main(int argc, char* argv[])
{
	if (argc != 4)
	{
		printf("Usage : DCDMergeSame.exe <dcd_files_dir> <same_thres> <space>\n");
		return -1;
	}
	string dcd_file_dir = argv[1];
	float same_thres = atof(argv[2]);
	string space = argv[3];
	bool in_luv_space = space == "LUV";
	vector<string> dcd_file_names;
	read_file_list(dcd_file_dir, dcd_file_names);
	ofstream save_merge_files[BIG_CLS];
	save_merge_files[0] = ofstream(dcd_file_dir + "0_thres" + argv[2] + "_" + space+ "_same.csv");
	save_merge_files[1] = ofstream(dcd_file_dir + "1_thres" + argv[2] + "_" + space + "_same.csv");
	save_merge_files[2] = ofstream(dcd_file_dir + "2_thres" + argv[2] + "_" + space + "_same.csv");
	for (string dcd_file_name : dcd_file_names) {
		string dcd_file_path = dcd_file_dir + dcd_file_name;

		/*string root_dir = "F:\\Project\\Commodity and Logo Recognization\\AI_JD\\crop_dataset\\";
		string dcd_file_path = root_dir + "show_set_new_fg_dcd_refine_8.txt";
		string save_merge_file_path = root_dir + "show_set_new_pairs.txt";*/
		ifstream dcd_file(dcd_file_path);
		vector<vector<DCDComponent> >dcd_descs;
		uchar rgb[3];
		vector<string> img_names;
		string img_name;
		int dominat_color_num, spatialCoherency;
		/*int pre = -1;
		bool first = true;*/
		while (dcd_file >> img_name >> dominat_color_num >> spatialCoherency)
		{
			/*if (first)
			{
				pre = dominat_color_num;
				first = false;
			}*/
			img_names.push_back(img_name.substr(0, img_name.rfind('.')));
			//assert(dominat_color_num == pre);
			//pre = dominat_color_num;
			vector<DCDComponent> dcd_desc;
			float percent_sum = 0;
			for (int i = 0; i < dominat_color_num; i++)
			{
				DCDComponent dcd;
				int percent;
				dcd_file >> percent;
				dcd.percent = percent / 100.0;
				dcd_file >> dcd.r >> dcd.g >> dcd.b;
				dcd.in_luv_space = in_luv_space;
				if (percent > 0)
				{
					dcd_desc.push_back(dcd);
					if (in_luv_space)
					{
						dcd.toLuv();
					}
				}
				percent_sum += dcd.percent;
			}
			for (auto& dcd : dcd_desc)
				dcd.percent /= percent_sum;
			stable_sort(dcd_desc.begin(), dcd_desc.end(), DCDComponent::cmp);
			dcd_descs.push_back(dcd_desc);
		}
		vector<bool> used(dcd_descs.size(), 0);
		vector<vector<int>> sames;
		vector<float> sim_scores;
		for (int i = 0; i < dcd_descs.size(); i++)
		{
			//if (used[i]) continue;
#ifdef FOR_TEST_MATCH
			if (img_names[i][5] != 't') continue;
#endif
			vector<int> same;
			same.push_back(i);
			//vector<float> intra_sim_scores;
			float sim_score = 0;
			for (int j = i + 1; j < dcd_descs.size(); j++)
			{
#ifdef FOR_TEST_MATCH
				if (img_names[j][5] == 't') continue;
#endif
				if (abs(int(dcd_descs[i].size() - dcd_descs[j].size())) > 1) continue;
				//if (used[j]) continue;
				float sim = calculate_dcd_similarity(dcd_descs[i], dcd_descs[j]);
				if (sim > same_thres)
				{
					same.push_back(j);
					//used[i] = used[j] = true;
					if (sim > sim_score) sim_score = sim;
					//intra_sim_scores.push_back(sim);
				}
			}
			if (same.size() >= 2)
			{
				sames.push_back(same);
				sim_scores.push_back(sim_score);
				//intra_sim_scores.push_back(sim_score);
				//sort(sames.begin(), sames.end(), [&intra_sim_scores](int x, int y) {return intra_sim_scores[x] > intra_sim_scores[y]; });
			}
		}
		vector<int> indics(sames.size());
		for (int j = 0;j < sames.size(); j++)
		{
			indics[j] = j;
		}
		sort(indics.begin(), indics.end(), [&sim_scores](int x, int y) {return sim_scores[x] > sim_scores[y];});
		int big_cls_id = dcd_file_name[0] - '0';
		ofstream& save_merge_file = save_merge_files[big_cls_id];
		for (int index:indics)
		{
			auto& same = sames[index];
			for (auto img_index : same)
			{
				save_merge_file << img_names[img_index] << ",";
			}
			
			save_merge_file<< sim_scores[index] << endl;
		}
	}
	for (int i = 0; i < BIG_CLS; i++)
	{
		save_merge_files[i].close();
	}
	return 0;
}