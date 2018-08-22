#include "util.h"
#include <queue>
#include <set>
#include <ctime>
#define COMPONENT_REMOVE_THRES1 0.03
#define COMPONENT_REMOVE_THRES2 0.13
//#define DETECT_BORDER
#define CENTER_R 0.05
Mat Gaussian_kernal(int kernel_size, float sigma)
{
	int m = kernel_size / 2;

	//Mat kernel(kernel_size, kernel_size, CV_32FC1,Scalar(1.0));
	Mat kernel(kernel_size, kernel_size, CV_32FC1);
	float s = 2 * sigma*sigma;
	for (int i = 0; i < kernel_size; i++)
	{
		for (int j = 0; j < kernel_size; j++)
		{
			int x = i - m, y = j - m;
			kernel.ptr<float>(i)[j] = exp(-(x*x + y*y) / s);
		}
	}
	Scalar all = sum(kernel);
	Mat normalK;
	kernel.convertTo(normalK, CV_32FC1, 1 / all[0]);
	return normalK;
}
int queryMergeSet(vector<int>& mergeSet, int query)
{
	if (query == mergeSet[query]) return query;
	return mergeSet[query] = queryMergeSet(mergeSet, mergeSet[query]);
}
void erode(int* closest, int img_w, int img_h, int winsize, int nc)
{
	Mat kernel = Gaussian_kernal(winsize, winsize / 2);
	//Mat kernel(winsize, winsize, CV_32FC1, Scalar(1.0));
	//cout << kernel << endl;
	float* clusterCount = new float[nc + 1];
	int* closestCpy = new int[img_w*img_h];
	memcpy(closestCpy, closest, img_w*img_h*sizeof(int));
	for (int i = 0; i < img_h; i++)
	{
		for (int j = 0; j < img_w; j++)
		{
			memset(clusterCount, 0, (nc + 1)*sizeof(float));
			int m = winsize / 2;
			for (int k = 0; k < winsize; k++)
			{
				int y = i + k - m;
				if (y<0 || y >= img_h) continue;
				for (int r = 0; r < winsize; r++)
				{
					int x = j + r - m;
					if (x<0 || x >= img_w) continue;
					int close_c = closestCpy[y * img_w + x];
					if (close_c == -1) close_c = nc;
					clusterCount[close_c] += kernel.at<float>(k, r);
				}
			}
			int max_c = nc;
			for (int k = 0; k < nc; k++)
			{
				if (clusterCount[k] > clusterCount[max_c])
				{
					max_c = k;
				}
			}
			if (max_c == nc) max_c = -1;
			closest[i * img_w + j] = max_c;
		}
	}
	delete[]clusterCount;
	delete[]closestCpy;
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
		if (postfix.compare("JPG") == 0 || postfix.compare("jpg") == 0 || postfix.compare("png") == 0 || postfix.compare("PNG") == 0) {
			img_names.push_back(filename);
		}

	} while (_findnext(hFile, &fileInfo) == 0);

	_findclose(hFile);    // 关闭搜索句柄

}
double cluster(int nClusters, float centroids[][3], int *closest,
	float *imdata, int imsize,
	unsigned char *quantImageAlpha)
{

	int     i, j, jmin;
	double  d1, d2, d3, dist, distmin, disttot = 0.0;
	float  *im1, *im2, *im3;
	int	imsize_msk;
	unsigned char *pAlpha;

	/* cluster */
	imsize_msk = 0;
	for (i = 0, im1 = imdata, im2 = imdata + 1, im3 = imdata + 2;
	i<imsize;
		i++, im1 += 3, im2 += 3, im3 += 3) {
		pAlpha = &quantImageAlpha[i];
		if (!quantImageAlpha || *pAlpha) {
			jmin = 0;
			distmin = FLT_MAX;
			for (j = 0; j < nClusters; j++) {
				d1 = *im1 - centroids[j][0];
				d2 = *im2 - centroids[j][1];
				d3 = *im3 - centroids[j][2];
				dist = d1*d1 + d2*d2 + d3*d3;
				if (dist < distmin) {
					jmin = j;
					distmin = dist;
				}
			}
			if (jmin < DSTMIN)
				closest[i] = jmin;
			disttot += distmin;
			imsize_msk++;
		}
	}

	//	return disttot/imsize;
	return disttot / imsize_msk;
} /* Cluster */
int bfs(int y, int x, Mat& img, Mat& imgLabel, int label)
{
	int img_w = img.cols;
	int img_h = img.rows;
	queue<int> points_queue;
	Mat in_queue(img_h, img_w, CV_8UC1, Scalar(0));
	int start = y *img_w + x;
	int* imgData = (int*)img.data;
	int* labelData = (int*)imgLabel.data;
	int labelColor = imgData[start];
	points_queue.push(start);
	in_queue.data[start] = 1;
	int imsize = img_w * img_h;
	int componentSize = 0;
	while (!points_queue.empty())
	{
		int cur_index = points_queue.front();
		points_queue.pop();
		componentSize++;
		labelData[cur_index] = label;
		int next_index = cur_index - img_w;
		if (next_index >= 0 && imgData[next_index] == labelColor && !in_queue.data[next_index])
		{
			in_queue.data[next_index] = 1;
			points_queue.push(next_index);
		}
		next_index = cur_index + img_w;
		if (next_index < imsize&& imgData[next_index] == labelColor && !in_queue.data[next_index])
		{
			in_queue.data[next_index] = 1;
			points_queue.push(next_index);
		}
		next_index = cur_index - 1;
		if (cur_index % img_w && imgData[next_index] == labelColor && !in_queue.data[next_index])
		{
			in_queue.data[next_index] = 1;
			points_queue.push(next_index);
		}
		next_index = cur_index + 1;
		if (next_index % img_w&& imgData[next_index] == labelColor && !in_queue.data[next_index])
		{
			in_queue.data[next_index] = 1;
			points_queue.push(next_index);
		}
	}
	return componentSize;
}
Mat draw_component_map(Mat& component_map,int color_num)
{
	srand(time(0));
	vector<Vec3b> colors(color_num);
	for (int i = 0; i < color_num; i++) {
		colors[i] = Vec3b(rand() % 255, rand() % 255, rand() % 255);
	}
	int imsize = component_map.cols*component_map.rows;
	Mat show_map(component_map.size(), CV_8UC3);
	for (int i = 0; i < component_map.rows; i++)
	{
		for (int j = 0; j < component_map.cols; j++)
		{
			int color_index = component_map.at<int>(i, j);
			if (color_index == -1)
				show_map.at<Vec3b>(i, j) = Vec3b(0,0,0);
			else
				show_map.at<Vec3b>(i, j) = colors[color_index];

		}
	}
	return show_map;
}
/*offset(left_Line,right_line,top_line*/
void edge_connected_component(Mat& img, Mat& imgLabel, Vec4i offset, vector<EdgeComponent>& components)
{
	//dominant_color_map = Mat(4, 4, CV_8UC1, Scalar(0));

	int img_w = img.cols;
	int img_h = img.rows;
	imgLabel = Mat(img_h, img_w, CV_32SC1, Scalar(-1));
	
	for (int k = 2; k < 4; k++)
	{
		int i = offset[k];
		for (int j = offset[0]; j <= offset[1]; j++)
		{
			if (imgLabel.at<int>(i, j) == -1)
			{
				int c = bfs(i, j, img, imgLabel, components.size());
				components.push_back(EdgeComponent(components.size(), img.at<int>(i, j), c));
			}
		}
	}
	for (int k = 0; k < 2; k++)
	{
		int j = offset[k];
		for (int i = offset[2]; i <= offset[3]; i++)
		{
			if (imgLabel.at<int>(i, j) == -1)
			{
				int c = bfs(i, j, img, imgLabel, components.size());
				components.push_back(EdgeComponent(components.size(), img.at<int>(i, j), c));
			}
		}
	}
	for (int k = 2; k < 4; k++)
	{
		int i = offset[k];
		for (int j = offset[0]; j <= offset[1]; j++)
		{
			components[imgLabel.at<int>(i, j)].edge_length++;
		}
	}
	for (int k = 0; k < 2; k++)
	{
		int j = offset[k];
		for (int i = offset[2]; i <= offset[3]; i++)
		{
			components[imgLabel.at<int>(i, j)].edge_length++;
		}
	}
#ifdef CENTER_R
	int center_x = (offset[0] + offset[1]) / 2;
	int center_y = (offset[2] + offset[3]) / 2;
	int w = offset[1] - offset[0] + 1, h = offset[3] - offset[2] + 1;
	int center_left = center_x - w * CENTER_R;
	int center_right = center_x + w * CENTER_R;
	int center_top = center_y - h * CENTER_R;
	int center_bottom = center_y + h * CENTER_R;

	//vector<Vec4i> boxes(components.size(), Vec4i(offset[1], offset[0], offset[3], offset[2]));
	vector<bool> pass_center(components.size(),false);
	for (int i = offset[2]; i <= offset[3]; i++)
		for (int j = offset[0]; j <= offset[1]; j++)
		{
			int component_id = imgLabel.at<int>(i, j);
			if (component_id < 0)continue;
			if (j >= center_left && j <= center_right && i >= center_top&&i <= center_bottom)
				pass_center[component_id] = true;

			//if (boxes[component_id][0] > j) boxes[component_id][0] = j;
			//if (boxes[component_id][1] < j) boxes[component_id][1] = j;
			//if (boxes[component_id][2] > i) boxes[component_id][2] = i;
			//if (boxes[component_id][3] < i) boxes[component_id][3] = i;
			//if (j >= center_left && j <= center_right && i >= center_top&&i <= center_bottom)
			//	pass_center[component_id] = true;
		}
	int imsize = w*h;
	int imlength = 2 * w + 2 * h;
	for (int i = 0; i < components.size(); i++)
	{
		if (pass_center[i] && components[i].size < 0.15 * imsize && components[i].edge_length < 0.1 * imlength)
		{
			components[i].must_keep = true;
			/*int component_area = (boxes[i][1] - boxes[i][0] + 1)*(boxes[i][3] - boxes[i][2] + 1);
			if (components[i].size > 0.7 * component_area)
				components[i].must_keep = true;*/
		}
	}
#endif
}

void merge_same_edge_components(int dominant_color_num, Mat& component_map, vector<EdgeComponent>& edge_components)
{
	//same color merge
	vector<EdgeComponent> c_components(dominant_color_num);
	vector<vector<int> > merge_set(dominant_color_num);
	vector<EdgeComponent> merged_components;
	vector<int> ids(edge_components.size());
	for (auto ec : edge_components)
	{
		if (ec.must_keep)
		{
			ids[ec.id] = merged_components.size();
			ec.id = merged_components.size();
			merged_components.push_back(ec);
		}
		else {
			EdgeComponent &c_ec = c_components[ec.color];
			merge_set[ec.color].push_back(ec.id);
			c_ec.edge_length += ec.edge_length;
			c_ec.size += ec.size;
			c_ec.color = ec.color;
		}
	}
	//edge_components.clear();
	int index = merged_components.size();
	for (int i = 0; i < dominant_color_num;i++)
	{
		EdgeComponent& c_ec = c_components[i];
		if (merge_set[i].size()) {
			for (int old_id : merge_set[i])
			{
				ids[old_id] = index;
			}
			c_ec.id = index++;
			merged_components.push_back(c_ec);
		}
	}
	edge_components = merged_components;
	int imsize = component_map.cols*component_map.rows;
	int* mapData = (int*)component_map.data;
	for (int i = 0; i < imsize;i++)
	{
		if(mapData[i] >= 0)
			mapData[i] = ids[mapData[i]];
	}
}
void merge_close_edge_components(float m_Centroids[][3], int dominant_color_num, Mat& component_map, vector<EdgeComponent>& edge_components)
{
	//same color merge
	int component_num = edge_components.size();
	double** dists = new double*[component_num];
	for (int i = 0; i < component_num; i++)
	{
		dists[i] = new double[component_num];
	}

	double distthr = 100;
	/* while two closest colours are closer than DISTTHR,
	merge the closest pair */
	int left_component = component_num;
	vector<bool> invalid(component_num,false);
	vector<int>merge_set(component_num);
	for (int i = 0; i < component_num; i++)
	{
		merge_set[i] = i;
		if (edge_components[i].must_keep)
		{
			invalid[i] = true;
			left_component--;
		}
	}
	double  distmin = 0.0;
	do {
		double  w1min, w2min;
		int     jamin, jbmin;
		/* initialise distance table */
		for (int ja = 0; ja < component_num; ja++) {
			if (invalid[ja]) continue;
			for (int jb = 0; jb < ja; jb++) {
				if (invalid[jb]) continue;
				double d1 = m_Centroids[ja][0] - m_Centroids[jb][0];
				double d2 = m_Centroids[ja][1] - m_Centroids[jb][1];
				double d3 = m_Centroids[ja][2] - m_Centroids[jb][2];
				dists[ja][jb] = d1*d1 + d2*d2 + d3*d3;
			}
		}
		/* find two closest colours */
		distmin = FLT_MAX;
		jamin = 0;
		jbmin = 0;
		for (int ja = 0; ja < component_num; ja++)
		{
			if (invalid[ja]) continue;
			for (int jb = 0; jb<ja; jb++) {
				if (invalid[jb]) continue;
				if (dists[ja][jb] < distmin) {
					distmin = dists[ja][jb];
					jamin = ja;
					jbmin = jb;
				}
			}
		}
		if (distmin > distthr)
			break;

		/* merge two closest colours */
		w1min = edge_components[jamin].size;
		w2min = edge_components[jbmin].size;
		int ja_color = edge_components[jamin].color;
		int jb_color = edge_components[jbmin].color;
		m_Centroids[ja_color][0] = m_Centroids[jb_color][0] = (w1min*m_Centroids[ja_color][0] +
			w2min*m_Centroids[jb_color][0]) / (w1min + w2min);
		m_Centroids[ja_color][1] = m_Centroids[jb_color][1] = (w1min*m_Centroids[ja_color][1] +
			w2min*m_Centroids[jb_color][1]) / (w1min + w2min);
		m_Centroids[ja_color][2] = m_Centroids[jb_color][2] = (w1min*m_Centroids[ja_color][2] +
			w2min*m_Centroids[jb_color][2]) / (w1min + w2min);
		edge_components[jbmin].size += edge_components[jamin].size;
		edge_components[jbmin].edge_length += edge_components[jamin].edge_length;
		invalid[jamin] = true;
		left_component--;
		merge_set[queryMergeSet(merge_set, jamin)] = queryMergeSet(merge_set, jbmin);
	} while (left_component > 1 && distmin < distthr);
	int index = 0;
	vector<EdgeComponent> new_components;
	for (int i = 0; i < component_num; i++)
	{
		if (!invalid[i] || edge_components[i].must_keep)
		{
			edge_components[i].id = index;
			new_components.push_back(edge_components[i]);
			index++;
		}
	}
	int imsize = component_map.cols*component_map.rows;
	int *data = (int*)component_map.data;
	for (int i = 0; i < imsize; i++)
	{
		if (data[i] >= 0)
		{
			data[i] = edge_components[queryMergeSet(merge_set, data[i])].id;
		}
	}
	/*edge_components.resize(new_components.size());
	for (int i = 0; i < new_components.size();i++)
	{
		edge_components[i] = new_components[i];
	}*/
	edge_components = new_components;
	for (int i = 0; i < component_num; i++)
	{
		delete [] dists[i];
	}
	delete[] dists;
}
Vec4i proccessBorder(Mat& component_map,Vec4i& old_border, EdgeComponent& biggestComponent, int imsize, int img_borders_length)
{
	int w = component_map.cols, h = component_map.rows;
	if (biggestComponent.edge_length > 0.9 * img_borders_length &&  biggestComponent.size < 0.12 * imsize)
	{
		int left = old_border[0];
		int mid_h = h / 2;
		while (component_map.at<int>(mid_h, left) != biggestComponent.id) mid_h++;
		while (component_map.at<int>(mid_h, left) == biggestComponent.id && left < w / 3)
		{
			left++;
		}
		if(component_map.at<int>(mid_h, left) == biggestComponent.id) return Vec4i(-1);
		int right = old_border[1];
		mid_h = h / 2;
		while (component_map.at<int>(mid_h, right) != biggestComponent.id) mid_h++; 
		while (component_map.at<int>(mid_h, right) == biggestComponent.id && right > 2 * w / 3)
		{
			right--;
		}
		if (component_map.at<int>(mid_h, right) == biggestComponent.id) return Vec4i(-1);
		int top = old_border[2];
		int mid_w = w / 2;
		while (component_map.at<int>(top, mid_w) != biggestComponent.id) mid_w++;
		while (component_map.at<int>(top, mid_w) == biggestComponent.id && top < h / 3)
		{
			top++;
		}
		if (component_map.at<int>(top, mid_w) == biggestComponent.id) return Vec4i(-1);
		int bottom = old_border[3];
		mid_w = w / 2;
		while (component_map.at<int>(bottom, mid_w) != biggestComponent.id) mid_w++;
		while (component_map.at<int>(bottom, mid_w) == biggestComponent.id && bottom > 2 * h / 3)
		{
			bottom--;
		}
		if (component_map.at<int>(bottom, mid_w) == biggestComponent.id) return Vec4i(-1);
		Vec4i border_size(left - old_border[0], old_border[1] - right, top - old_border[2], old_border[3] - bottom);
		float h_ratio = float(border_size[0]) / (border_size[0] + border_size[1]);
		float v_ratio = float(border_size[2]) / (border_size[2] + border_size[3]);
		float EPS = 0.09;
		if (abs(h_ratio - 0.5) < EPS && abs(v_ratio - 0.5) < EPS)
		{
			return Vec4i(left + 1, right - 1,
				top + 1, bottom - 1);
		}
	}
	return Vec4i(-1);
}
bool filterEdgeComponent(Mat& component_map,Vec4i& border, vector<EdgeComponent>& edge_components, vector<uchar>& keepEdgeComponent)
{
	int imsize = component_map.cols * component_map.rows;
	int edge_component_num = edge_components.size();
	vector<EdgeComponent> simple_edge_components = edge_components;
	vector<EdgeComponent>::iterator it = simple_edge_components.begin();
	while (it < simple_edge_components.end())
	{
		EdgeComponent ec = *it;
		if (ec.size < 40)
		{
			//img_edge_length -= ec.edge_length;
			it = simple_edge_components.erase(it);
		}
		else {
			it++;
		}
	}
	sort(simple_edge_components.begin(), simple_edge_components.end());
	int left_num = simple_edge_components.size();
	int tol_length = 0;
	int tolComponentSize = 0;
	for (auto ec : simple_edge_components)
	{
		tol_length += ec.edge_length;
		tolComponentSize += ec.size;
	}
	Vec4i new_border(-1);
#ifdef DETECT_BORDER
	new_border = proccessBorder(component_map, border, simple_edge_components[left_num - 1], imsize, tol_length);
#endif
	if (new_border[0] > 0)
	{
		border = new_border;
		return true;
	}
	int i = 0;
	while (tolComponentSize > 0.96 * imsize)
	{
		tolComponentSize -= simple_edge_components[i].size;
		keepEdgeComponent[simple_edge_components[i].id] = 1;
		i++;
	}
	float top_length = 0;
	int top_k = 0;
	while (top_length < tol_length * 0.9)
	{
		top_length += simple_edge_components[left_num - top_k - 1].edge_length;
		top_k++;
	}
	float keep_proportion;
	if (top_k <= 2)
	{
		keep_proportion = 0.24;
	}
	else if (top_k <= 4)
	{
		keep_proportion = 0.2;
	}
	else if (top_k <= 7)
	{
		keep_proportion = 0.15;
	}
	else {
		keep_proportion = 0.1;
	}
	float keep_edge_length = 0;
	for (auto ec : simple_edge_components)
	{
		if (ec.must_keep)
		{
			keepEdgeComponent[ec.id] = 1;
			keep_edge_length += ec.edge_length;
		}
	}
	i = 0;
	while (1)
	{
		if (!simple_edge_components[i].must_keep)
		{
			keep_edge_length += simple_edge_components[i].edge_length;
			if(keep_edge_length < keep_proportion * tol_length)
				keepEdgeComponent[simple_edge_components[i].id] = 1;
			else break;
		}
		i++;
	}
	return false;
	//int top_k = 4;
	//float least_proportion = 0.76;
	//float step = 0.04;
	//for (int k = 1; k <= top_k; k++)
	//{
	//	if (left_num > k)
	//	{
	//		top_length += simple_edge_components[left_num - k].edge_length;
	//		float proportion = (float)top_length / img_edge_length;
	//		if (proportion > least_proportion)
	//		{
	//			for (int i = 0; i < left_num - k; i++)
	//			{
	//				keepEdgeComponent[simple_edge_components[i].id] = 1;
	//			}
	//			return;
	//		}
	//	}
	//	//least_proportion += step;
	//}
}

int twoPass(Mat& img, Mat& imgLabel, uchar val, int minLabel)
{
	/*img = Mat(3, 3, CV_8UC1, Scalar(255));
	img.data[4] = img.data[8] = 0;*/
	int img_w = img.cols;
	int img_h = img.rows;
	if (imgLabel.empty())
	{
		imgLabel = Mat(img_h, img_w, CV_32SC1, Scalar(-1));
	}
	int curLabel = minLabel;
	vector<int> mergeSet;
	for (int i = 0; i < img_h; i++)
	{
		for (int j = 0; j < img_w; j++)
		{
			if (img.at<uchar>(i, j) == val)
			{
				int leftLabel = j > 0 ?imgLabel.at<int>(i, j - 1) : -1;
				int upLabel = i > 0 ? imgLabel.at<int>(i - 1, j) : -1;
				int &ijLabel = imgLabel.at<int>(i, j);
				if (leftLabel < minLabel && upLabel < minLabel)
				{
					mergeSet.push_back(curLabel - minLabel);
					ijLabel = curLabel++;
				}
				else if (leftLabel >= minLabel && upLabel >= minLabel)
				{
					if (leftLabel < upLabel)
					{
						int mLabel = queryMergeSet(mergeSet, leftLabel - minLabel) + minLabel;
						mergeSet[upLabel - minLabel] = mLabel - minLabel;
						ijLabel = mLabel;
					}
					else {
						int mLabel = queryMergeSet(mergeSet, upLabel - minLabel) + minLabel;
						mergeSet[leftLabel - minLabel] = mLabel - minLabel;
						ijLabel = mLabel;
					}
				}
				else if (leftLabel >= minLabel)
				{
					ijLabel = queryMergeSet(mergeSet, leftLabel - minLabel) + minLabel;
				}
				else {
					ijLabel = queryMergeSet(mergeSet, upLabel - minLabel) + minLabel;
				}
			}
		}
	}

	for (int i = 0; i < img_h; i++)
	{
		for (int j = 0; j < img_w; j++)
		{
			int &ijLabel = imgLabel.at<int>(i, j);
			if (ijLabel >= minLabel)
			{
				ijLabel = queryMergeSet(mergeSet, ijLabel - minLabel) + minLabel;
			}
		}
	}
	vector<int> appear(mergeSet.size(), 0);
	for (int label : mergeSet)
	{
		appear[label] = 1;
	}
	int counter = 0;
	for (int i = 0; i < mergeSet.size(); i++)
	{
		if (appear[i] == 1)
		{
			appear[i] = counter++;
		}
	}
	for (int i = 0; i < img_h; i++)
	{
		for (int j = 0; j < img_w; j++)
		{
			int &ijLabel = imgLabel.at<int>(i, j);
			if (ijLabel >= minLabel)
			{
				ijLabel = appear[ijLabel - minLabel] + minLabel;
			}
		}
	}
	return counter;
}
bool RectIntersected(Vec4i& r1,Vec4i& r2)
{
	int left_most = r1[0] < r2[0] ? r2[0] : r1[0];
	int	right_most = r1[2] > r2[2] ? r2[2] : r1[2];
	if (right_most < left_most)
		return false;
	int top_most = r1[1] < r2[1] ? r2[1] : r1[1];
	int	bottom_most = r1[3] > r2[3] ? r2[3] : r1[3];
	if (bottom_most < top_most)
		return false;
	return true;
}
bool RectContains(Vec4i& big, Vec4i& small)
{
	return big[0] <= small[0] && big[1] <= small[1] && big[2] >= small[2] && big[3] >= small[3];
}
Mat removeIsolatedComponent(Mat& mask, Mat& imgLabel, int labelCount)
{
	vector<double> componentScore(labelCount, 0);
	int kernelSize = 9;
	Mat kernel = Gaussian_kernal(kernelSize, kernelSize / 2);
	int img_w = mask.cols;
	int img_h = mask.rows;
	vector<Vec4i> componentRects(labelCount,Vec4i(-1));
	double tolScore = 0;
	for (int i = 0; i < img_h; i++)
	{
		int k = (i * kernelSize) / img_h;
		for (int j = 0; j < img_w; j++)
		{
			int label = imgLabel.at<int>(i, j);
			if (label >= 0)
			{
				float score = kernel.at<float>(k, (j * kernelSize) / img_w);
				componentScore[label] += score;
				tolScore += score;
				if (componentRects[label][0] < 0)
				{
					componentRects[label] = Vec4i(j, i, j, i);
				}
				else {
					if (componentRects[label][0] > j)
					{
						componentRects[label][0] = j;
					}
					else if (componentRects[label][2] < j)
					{
						componentRects[label][2] = j;
					}
					if (componentRects[label][1] > i)
					{
						componentRects[label][1] = i;
					}
					else if (componentRects[label][3] < i)
					{
						componentRects[label][3] = i;
					}
				}
			}
		}
	}
	vector<uchar> removed_compoent(labelCount, 0);
	/*for (int i = 0; i < labelCount; i++)
	{
	if (componentScore[i] / tolScore < COMPONENT_REMOVE_THRES)
	{
	removed_compoent[i] = 1;
	}
	}*/
	double max_score = 0;
	int max_label = -1;
	for (int i = 0; i < labelCount; i++)
	{
		if (max_score < componentScore[i])
		{
			max_score = componentScore[i];
			max_label = i;
		}
	}
	for (int i = 0; i < labelCount; i++)
	{
		if (componentScore[i] < max_score * COMPONENT_REMOVE_THRES1) {
			removed_compoent[i] = 1;
		}
		else if (componentScore[i]  < max_score * COMPONENT_REMOVE_THRES2 && !RectContains(componentRects[max_label], componentRects[i]))
		{
			removed_compoent[i] = 1;
		}
	}
	assert(max_score <= tolScore);
	Mat BiggestCompoentMask = mask.clone();
	for (int i = 0; i < img_h; i++)
	{
		for (int j = 0; j < img_w; j++)
		{
			int label = imgLabel.at<int>(i, j);
			if (label >= 0 && removed_compoent[label])
			{
				mask.at<uchar>(i, j) = 0;
			}
			if (label != max_label)
			{
				BiggestCompoentMask.at<uchar>(i, j) = 0;
			}
		}
	}
	return BiggestCompoentMask;
}
Rect getMaskRect(const Mat& mask)
{
	int img_w = mask.cols;
	int img_h = mask.rows;
	int l = img_w, r = 0, u = img_h, d = 0;
	for (int i = 0; i < img_h; i++)
	{
		for (int j = 0; j < img_w; j++)
		{
			if (mask.at<uchar>(i, j))
			{
				if (i < u) u = i;
				if (i > d) d = i;
				if (j < l) l = j;
				if (j > r) r = j;
			}
		}
	}
	int keep_border = 4;
	if (l >= keep_border) l -= keep_border;
	if (u >= keep_border) u -= keep_border;
	if (r + keep_border < img_w) r = r + keep_border;
	if (d + keep_border < img_h) d = d + keep_border;
	return Rect(l, u, r - l + 1, d - u + 1);
}
void getForeground(const cv::Mat& img, const cv::Mat& mask, cv::Mat& result)
{
	//generate alpha image
	result = cv::Mat(img.size(), CV_8UC4);
	int imsize = img.cols * img.rows;
	for (int i = 0; i < imsize; i++)
	{
		if (mask.data[i])
			for (int j = 0; j < 3; j++) 
				result.data[4 * i + j] = img.data[3 * i + j];
		else
			for (int j = 0; j < 3; j++)
				result.data[4 * i + j] = 127;
		result.data[4 * i + 3] = mask.data[i];
	}
}
void get_component_stds(const Mat& origin_img, Mat& components_map, vector<EdgeComponent>& components, vector<Vec3f> &component_stds)
{
	int nc = components.size();
	Mat blurredImg;
	medianBlur(origin_img, blurredImg, 3);
	vector<Vec3f> component_means(nc);
	int imsize = origin_img.cols*origin_img.rows;
	int* mapData = (int*)components_map.data;
	for (int i = 0; i < imsize; i++)
	{
		int label = mapData[i];
		if (label >= 0)
		{
			for (int j = 0; j < 3; j++)
				component_means[label][j] += blurredImg.data[3 * i + j];
		}
	}
	for (int i = 0; i < nc; i++)
	{
		for (int j = 0; j < 3; j++)
			component_means[i][j] /= components[i].size;
	}
	for (int i = 0; i < imsize; i++)
	{
		int label = mapData[i];
		if (label >= 0)
		{
			for (int j = 0; j < 3; j++)
			{
				int tmp = blurredImg.data[3 * i + j] - component_means[label][j];
				component_stds[label][j] += tmp*tmp;
			}
		}
	}
	for (int i = 0; i < nc; i++)
	{
		for (int j = 0; j < 3; j++) {
			component_stds[i][j] = sqrt(component_stds[i][j] / components[i].size);
		}
	}
}
void edge_erode(Mat& src, Mat& dst, int kernal_size)
{
	int img_w = src.cols;
	int img_h = src.rows;
	Mat tmp = src.clone();
	for (int i = 0; i < img_h; i++)
	{
		for (int j = 0; j < img_w; j++)
		{
			if (src.at<uchar>(i, j) == 0) continue;
			int m = kernal_size / 2;
			for (int k = 0; k < kernal_size; k++)
			{
				int y = i + k - m;
				if (y < 0 || y >= img_h) continue;
				for (int r = 0; r < kernal_size; r++)
				{
					int x = j + r - m;
					if (x < 0 || x >= img_w) continue;
					if (src.at<uchar>(y, x) == 0)
					{
						tmp.at<uchar>(i, j) =  0;
					}
				}
			}
			
		}
	}
	dst = tmp;
}
void edge_dilate(Mat& src, Mat& dst, int kernal_size)
{
	int img_w = src.cols;
	int img_h = src.rows;
	Mat tmp = src.clone();
	for (int i = 0; i < img_h; i++)
	{
		for (int j = 0; j < img_w; j++)
		{
			if (src.at<uchar>(i, j) == 255) continue;
			int m = kernal_size / 2;
			for (int k = 0; k < kernal_size; k++)
			{
				int y = i + k - m;
				if (y < 0 || y >= img_h) continue;
				for (int r = 0; r < kernal_size; r++)
				{
					int x = j + r - m;
					if (x < 0 || x >= img_w) continue;
					if (src.at<uchar>(y, x) == 255)
					{
						tmp.at<uchar>(i, j) = 255;
					}
				}
			}
		}
	}
	dst = tmp;
}

void filterStopMap1(Mat& src, Mat& dst, Mat& components_map, int kernal_size)
{
	int img_w = src.cols;
	int img_h = src.rows;
	Mat tmp(src.size(), CV_8UC1);
	int m = kernal_size / 2;
	int kernal_area = kernal_size*kernal_size;
	for (int i = 0; i < img_h; i++)
	{
		for (int j = 0; j < img_w; j++)
		{
			tmp.at<uchar>(i, j) = 0;
			if (src.at<uchar>(i, j) > 0) {
				set<int> diff_label;
				int dot_counter = 0;
				int center_label = components_map.at<int>(i, j);
				if (center_label == -1)continue;
				int neg_count = 0;
				for (int k = 0; k < kernal_size; k++)
				{
					int y = i + k - m;
					if (y < 0 || y >= img_h) continue;
					for (int r = 0; r < kernal_size; r++)
					{
						int x = j + r - m;
						if (x < 0 || x >= img_w) continue;
						int label = components_map.at<int>(y, x);
						diff_label.insert(label);
						if (label == -1) neg_count++;
					}
				}
				/*for (int k = 0; k < 3; k++)
				{
					int y = i + k - 1;
					if (y < 0 || y >= img_h) continue;
					for (int r = 0; r < 3; r++)
					{
						int x = j + r - 1;
						if (x < 0 || x >= img_w) continue;
						if (src.at<uchar>(y, x))
						{
							dot_counter++;
						}
					}
				}*/
				if (diff_label.size() == 1)
				{
					tmp.at<uchar>(i, j) = 255;
				}
				/*else if (diff_label.size() == 2)
				{
					if (neg_count == 0)
					{
						//tmp.at<uchar>(i, j) = 255;
					}
				}
				else if (dot_counter >= 3)
				{
					//tmp.at<uchar>(i, j) = 255;
				}*/
				/*if (diff_label.size() < 3 && )
					tmp.at<uchar>(i, j) = 255;*/

			}
		}
	}
	dst = tmp;
}
void filterStopMap1(Mat& src, Mat& components_map, int kernal_size)
{
	int img_w = src.cols;
	int img_h = src.rows;
	int m = kernal_size / 2;
	int kernal_area = kernal_size*kernal_size;
	for (int i = 0; i < img_h; i++)
	{
		for (int j = 0; j < img_w; j++)
		{
			if (src.at<uchar>(i, j) > 0) {
				int center_label = components_map.at<int>(i, j);
				if (center_label == -1) src.at<uchar>(i, j) = 0;
			}
		}
	}
}

void stop_edge_connected_component(const Mat& origin_img, int dominant_color_num, Mat& dominant_color_map, Mat& components_map, Vec4i offset, vector<EdgeComponent>& components)
{
	int nc = components.size();
	int imsize = origin_img.cols*origin_img.rows;
	vector<Vec3f> component_stds(nc);
	//get_component_stds(origin_img, components_map, components, component_stds);

	//for (int i = 0;i)
	float limit_std = 12;
	vector<bool> stop_compoents(nc, true);
	for (int i = 0; i < nc; i++)
	{
		Vec3f component_std = component_stds[i];
		float avg_std = (component_std[0] + component_std[1] + component_std[2]) / 3;
		if (avg_std > limit_std || components[i].size < 0.01 * imsize || components[i].must_keep) {
			stop_compoents[i] = false;
		}
	}
	int* components_map_data = (int*)components_map.data;
	Mat grayImage;
	//GaussianBlur(origin_img, grayImage, Size(3, 3), 0, 0, BORDER_DEFAULT);
	cvtColor(origin_img, grayImage, COLOR_BGR2GRAY);
	GaussianBlur(grayImage, grayImage, Size(3, 3), 0, 0, BORDER_DEFAULT);
	Mat stop_map = grayImage.clone();

	for (int i = 0; i < imsize; i++)
	{
		if (components_map_data[i] == -1 || !stop_compoents[components_map_data[i]])
		{
			stop_map.data[i] = 0;
		}
	}
	int lowThreshold = 4;
	Canny(stop_map, stop_map, lowThreshold, lowThreshold * 3, 3);
	/*for (int i = 0; i < imsize; i++)
	{
	if(stop_map.data[i])
	}*/
	//imshow("stop_map", stop_map);
	filterStopMap1(stop_map, stop_map, components_map, 3);
	//filterStopMap2(stop_map, stop_map, components_map, components.size());
	int dilate_time = 1;
	//imshow("origin stop Map", stop_map);
	while (dilate_time--)
		edge_dilate(stop_map, stop_map, 9);
	edge_erode(stop_map, stop_map, 9);
	//filterStopMap1(stop_map, stop_map, components_map,3);
	//filterStopMap1( stop_map, components_map, 3);
	//imshow("Stop Map", stop_map);
	//imshow("Component map", draw_component_map(components_map, components.size()));
	waitKey(0);
	Mat tmp_dominant_color_map = dominant_color_map.clone();
	int* tdcm_Data = (int*)tmp_dominant_color_map.data;
	for (int i = 0; i < imsize; i++)
	{
		if (stop_map.data[i])
		{
			int color;
			if (components_map_data[i] == -1)
				color = components.size() + dominant_color_num;
			else color = components_map_data[i] + dominant_color_num;
			tdcm_Data[i] = color;
		}
	}
	components.clear();
	//imshow("Stop Map", stop_map);
	waitKey(0);
	edge_connected_component(tmp_dominant_color_map, components_map, offset, components);
	for (int k = 2; k < 4; k++)
	{
		int i = offset[k];
		for (int j = offset[0]; j <= offset[1]; j++)
		{
			if (stop_map.at<uchar>(i, j))
			{
				int& c = components[components_map.at<int>(i, j)].color;
				c = dominant_color_map.at<int>(i, j);
			}
		}   
	}
	for (int k = 0; k < 2; k++)
	{
		int j = offset[k];
		for (int i = offset[2]; i <= offset[3]; i++)
		{
			if (stop_map.at<uchar>(i, j))
			{
				int& c = components[components_map.at<int>(i, j)].color;
				c = dominant_color_map.at<int>(i, j);
			}
		}
	}
	//namedWindow("stop_map_dilate", CV_WINDOW_FREERATIO);
	//imshow("stop_map_dilate", stop_map);
	//waitKey(0);
}
