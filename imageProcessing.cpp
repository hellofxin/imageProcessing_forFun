#include <stdio.h>
#include "opencv2/opencv.hpp"
#include <math.h>
#include <map>

using namespace cv;
using namespace std;

#define WIN_NAME_WIN_CONTROL	"winControl"
#define KSIZE					"ksize"
#define CANNY_MIN				"cannyMin"
#define CANNY_MAX				"cannyMax"
#define MORPH_OP				"morphOp"
#define THRES_B_SIZE			"thresBSize"
#define THRES_MEAN_C			"thresMeanC"

struct PriorInfo{
	const char* name;
	int range;
	float scaleStart;
	float scaleEnd;
	float x;
	float y;
	float radiusMin;
	float radiusMax;
	PriorInfo* p;
};

PriorInfo priorInfo = {"template1", 60, 135, 45, 0, 0, 0, 0, NULL};

Mat g_imgSrc, g_imgDst, g_imgMask, g_imgRound, g_imgAnnular,
	g_imgBlur, g_imgCanny, g_imgMorph, g_imgThreshod;

const char* g_imgName[] = {"./pic/src1.jpg"};

map<char*, char*> g_mapWinName;
map<char*, char*> g_mapTrackbarName;

/* ksize */
static int g_nTrackbarValue0 = 1;
static const int g_nTrackbarCount0 = 25;
/* canny min threshold */
static int g_nTrackbarValue1 = 100;
static const int g_nTrackbarCount1 = 500;
/* canny max threshold */
static int g_nTrackbarValue2 = 215;
static const int g_nTrackbarCount2 = 1000;
/* morphology */
static int g_nTrackbarValue3 = 3;
static const int g_nTrackbarCount3 = 8;
/* adaptive threshold */
static int g_nTrackbarValue4 = 20;
static const int g_nTrackbarCount4 = 100;
static int g_nTrackbarValue5 = 1100;
static const int g_nTrackbarCount5 = 3000;
/* get good match */
static int g_nTrackbarValue6 = 250;
static const int g_nTrackbarCount6 = 500;

/* ksize */
static int& g_nKSize = g_nTrackbarValue0;
/* canny min/max threshold */
static int& g_nCannyMinThreshold = g_nTrackbarValue1;
static int& g_nCannyMaxThreshold = g_nTrackbarValue2;
/* morphology */
static int& g_nMorphologyOption = g_nTrackbarValue3;
/* adaptive threshold */
static int& g_nThresholdBSize = g_nTrackbarValue4;
static int& g_nThresholdMeanC = g_nTrackbarValue5;
/* get good match */
static int& g_nGoodMatchDistance = g_nTrackbarValue6;

VideoCapture videoCapture;
Mat frame;
Mat imgSample, imgTempate, imgDst;

void initWinTrackbar();
void initImg();
void initCamera(const char* winname);
void init();
void go();

void registration(Mat& imgsrc, Mat& imgdst, Mat& imgTemplate);
/* ksize */
void trackbarCallback0(int trackbarValue, void* userdata);
void trackbar0(int value, Mat& userdataimg, Mat& imgdst);
/* canny */
void trackbarCallback1(int trackbarValue, void* userdata);
void trackbar1(int value, Mat& userdataimg, Mat& imgdst);
void trackbarCallback2(int trackbarValue, void* userdata);
void trackbar2(int value, Mat& userdataimg, Mat& imgdst);
/* morphology */
void trackbarCallback3(int trackbarValue, void* userdata);
void trackbar3(int value, Mat& userdataimg, Mat& imgdst);
/* threshold */
void trackbarCallback4(int trackbarValue, void* userdata);
void trackbar4(int value, Mat& userdataimg, Mat& imgdst);
void trackbarCallback5(int trackbarValue, void* userdata);
void trackbar5(int value, Mat& userdataimg, Mat& imgdst);
/* good match */
void trackbarCallback6(int trackbarValue, void* userdata);
void trackbar6(int value, Mat& userdataimg);
/**/
void mouseCallback(int event, int x, int y, int flags, void* userdata);

int main()
{
	init();
    return 0;
}

void initWinTrackbar(){
	g_imgSrc = imread("./pic/src1.jpg", IMREAD_GRAYSCALE);
	CV_Assert(g_imgSrc.data);
	g_imgDst = g_imgSrc.clone();
	g_imgMask = g_imgSrc.clone();
	g_imgRound = g_imgSrc.clone();
	g_imgBlur = g_imgSrc.clone();
	g_imgCanny = g_imgSrc.clone();
	g_imgThreshod = g_imgSrc.clone();
	/* win name *
	g_mapWinName.insert(pair<char*, char*>("winControl", "winControl"));
	g_mapWinName.insert(pair<char*, char*>("imgSrc", "imgSrc"));
	g_mapWinName.insert(pair<char*, char*>("imgDst", "imgDst"));
	g_mapWinName.insert(pair<char*, char*>("imgBlur", "imgBlur"));
	g_mapWinName.insert(pair<char*, char*>("imgCanny", "imgCanny"));
	g_mapWinName.insert(pair<char*, char*>("imgThreshold", "imgThreshold"));
	/* trackbar name 
	g_mapTrackbarName.insert(pair<char*, char*>("ksize", "ksize"));
	g_mapTrackbarName.insert(pair<char*, char*>("cannyMin", "cannyMin"));
	g_mapTrackbarName.insert(pair<char*, char*>("cannyMax", "cannyMax"));
	g_mapTrackbarName.insert(pair<char*, char*>("morphOp", "morphOp"));
	g_mapTrackbarName.insert(pair<char*, char*>("thresBSize", "thresBSize"));
	g_mapTrackbarName.insert(pair<char*, char*>("thresMeanC", "thresMeanC"));

	/**/
	namedWindow(WIN_NAME_WIN_CONTROL, WINDOW_NORMAL);
	cvResizeWindow(WIN_NAME_WIN_CONTROL, 400, 600);

	/* blur */
	createTrackbar(KSIZE, WIN_NAME_WIN_CONTROL,
		&g_nTrackbarValue0, g_nTrackbarCount0, trackbarCallback0, (void*)&g_imgSrc);
	/* canny */
	createTrackbar(CANNY_MIN, WIN_NAME_WIN_CONTROL,
		&g_nTrackbarValue1, g_nTrackbarCount1, trackbarCallback1, (void*)&g_imgBlur);
	createTrackbar(CANNY_MAX, WIN_NAME_WIN_CONTROL,
		&g_nTrackbarValue2, g_nTrackbarCount2, trackbarCallback2, (void*)&g_imgBlur);
	/* morphology */
	createTrackbar(MORPH_OP, WIN_NAME_WIN_CONTROL,
		&g_nTrackbarValue3, g_nTrackbarCount3, trackbarCallback3, (void*)&g_imgCanny);
	/* adaptive threshold */
	createTrackbar(THRES_B_SIZE, WIN_NAME_WIN_CONTROL,
		&g_nTrackbarValue4, g_nTrackbarCount4, trackbarCallback4, (void*)&g_imgBlur);
	createTrackbar(THRES_MEAN_C, WIN_NAME_WIN_CONTROL,
		&g_nTrackbarValue5, g_nTrackbarCount5, trackbarCallback5, (void*)&g_imgBlur);
	/**/
}

void initImg(){
	/* blur */
	trackbarCallback0(g_nTrackbarValue0, (void*)&g_imgSrc);
	/* canny *
	trackbarCallback1(g_nTrackbarValue1, (void*)&g_imgBlur);
	/* morphology *
	trackbarCallback3(g_nTrackbarValue3, (void*)&g_imgCanny);
	/* threshold */
	trackbarCallback4(g_nTrackbarValue4, (void*)&g_imgBlur);
	
}

void initCamera(const char* winname){
	char key;
	videoCapture.open(0);
	videoCapture >> frame;
	imshow(winname, frame);
	imgSample = frame.clone();
	imgTempate = frame.clone();
	namedWindow("winControl", WINDOW_GUI_NORMAL);
	createTrackbar("goodMatch", "winControl", &g_nTrackbarValue6, g_nTrackbarCount6, trackbarCallback6, 0);
	setMouseCallback(winname, mouseCallback);
	while(1){
		videoCapture >> frame;
		imshow(winname, frame);
		key = waitKey(50);
		if( key == 'w' ){
			break;
		}
		else if( key == 's' ){
			imwrite("./pic/sample.jpg", frame);
			imshow("sample", frame);
		}
		else if( key == 't' ){
			imwrite("./pic/template.jpg", frame);
			imshow("template", frame);
		}
		else if( key == 'r' ){
			go();
		}
	}
}

void init(){
	initWinTrackbar();
	initImg();
	initCamera("winShot");
}

void registration(Mat& imgsrc, Mat& imgdst, Mat& imgTemplate){
	cout<<"registration"<<endl;
	Mat imgShow;
	Ptr<ORB> orb = ORB::create();
	vector<KeyPoint> vKeyPoint1, vKeyPoint2;
	vector<vector<KeyPoint>> vvKeyPoint1, vvKeyPoint2;
	Mat imgDescriptor1, imgDescriptor2;

	Ptr<DescriptorMatcher> descriptorMatcher = DescriptorMatcher::create(DescriptorMatcher::BRUTEFORCE);
	vector<DMatch> vDMatch, vDMatchGood;
	vector<vector<DMatch>> vvDMatch;
	/**/
	vector<Point2f> vpGoodTemplate, vpGoodSample;
	Mat imgHSampleToTemplate, imgHTemplateToSample;
	vector<Point2f> vpCornerTemplate(4), vpCornerSample(4);
	Point2f offset(imgTemplate.cols - 1, 0);
	Scalar scalar(255);
	/**
	vKeyPoint1.resize(1000);
	vKeyPoint2.resize(1000);
	imgDescriptor1 = imgsrc.clone();
	imgDescriptor2 = imgsrc.clone();
	/* 特征匹配 */
	orb->detectAndCompute(imgTemplate, Mat(), vKeyPoint1, imgDescriptor1);
	orb->detectAndCompute(imgsrc, Mat(), vKeyPoint2, imgDescriptor2);
	descriptorMatcher->match(imgDescriptor1, imgDescriptor2, vDMatch);
	sort(vDMatch.begin(), vDMatch.end());
	for(int i = 0; i < vDMatch.size()*g_nGoodMatchDistance*0.001; i++){
		vDMatchGood.push_back(vDMatch[i]);
		vpGoodTemplate.push_back(vKeyPoint1[vDMatchGood[i].queryIdx].pt);
		vpGoodSample.push_back(vKeyPoint2[vDMatchGood[i].trainIdx].pt);
	}
	drawMatches(imgTemplate, vKeyPoint1, imgsrc, vKeyPoint2, vDMatchGood, imgShow);
	//imshow("registration drawMatches", imgShow);
	cout << "registration drawMatches" << endl;
	/* 透视变换 sample to template */
	imgHSampleToTemplate = findHomography(vpGoodSample, vpGoodTemplate, RANSAC);
	warpPerspective(imgsrc, imgdst, imgHSampleToTemplate, Size(imgTemplate.cols, imgTemplate.rows));
	imshow("registration", imgdst);
	cout << "registration warpPerspective" << endl;
	/* 目标寻找 template to sample */
	imgHTemplateToSample = findHomography(vpGoodTemplate, vpGoodSample, RANSAC);
	vpCornerTemplate[0] = Point2f(0, 0);
	vpCornerTemplate[1] = Point2f(imgTemplate.cols - 1, 0);
	vpCornerTemplate[2] = Point2f(imgTemplate.cols - 1, imgTemplate.rows - 1);
	vpCornerTemplate[3] = Point2f(0, imgTemplate.rows - 1);
	perspectiveTransform(vpCornerTemplate, vpCornerSample, imgHTemplateToSample);
	line(imgShow, vpCornerSample[0] + offset, vpCornerSample[1] + offset, scalar);
	line(imgShow, vpCornerSample[1] + offset, vpCornerSample[2] + offset, scalar);
	line(imgShow, vpCornerSample[2] + offset, vpCornerSample[3] + offset, scalar);
	line(imgShow, vpCornerSample[3] + offset, vpCornerSample[0] + offset, scalar);
	/* 从template 到 sample 透视后，template 中的点P(x, y)映射为对应坐标点P'(x', y')，
	因此，可以在显示图像上的坐标点P'(x', y')处，将其灰度值赋值为template的P(x, y)处的灰度值 */
	circle(imgShow, vpCornerSample[0] + offset, 2, scalar);
	circle(imgShow, vpCornerSample[1] + offset, 4, scalar);
	circle(imgShow, vpCornerSample[2] + offset, 6, scalar);
	circle(imgShow, vpCornerSample[3] + offset, 8, scalar);
	imshow("registration 目标寻找，在sample中找template的四个角", imgShow);
	cout << "registration 目标寻找，在sample中找template的四个角" << endl;
	/**/
}

void go() {
	imgSample = imread("./pic/sample.jpg", IMREAD_GRAYSCALE);
	imgTempate = imread("./pic/template.jpg", IMREAD_GRAYSCALE);
	if (imgSample.empty() || imgTempate.empty()) {
		imgSample = imread("./pic/sample1.jpg", IMREAD_GRAYSCALE);
		imgTempate = imread("./pic/template1.jpg", IMREAD_GRAYSCALE);
		cout << "read image for the first time" << endl;
	}
	imshow("sample", imgSample);
	imshow("template", imgTempate);
	CV_Assert(!imgSample.empty());
	CV_Assert(!imgTempate.empty());
	CV_Assert(imgSample.type() == CV_8UC1);
	CV_Assert(imgTempate.type() == CV_8UC1);
	registration(imgSample, imgDst, imgTempate);
}


/* ksize */
void trackbarCallback0(int trackbarValue, void* userdata){
	/* ksize */
	cout << "trackbarCallback0" << endl;
	Mat& userDataImg = *(Mat*)userdata;
	trackbar0(trackbarValue, userDataImg, g_imgBlur);
}
void trackbar0(int value, Mat& userdataimg, Mat& imgdst){
	int nKSize = value * 2 + 1;
	medianBlur(userdataimg, imgdst, nKSize);
	imshow("imgBlur", imgdst);
}

/* canny */
void trackbarCallback1(int trackbarValue, void* userdata){
	/* canny min */
	cout << "trackbarCallback1" << endl;
	Mat& userDataImg = *(Mat*)userdata;
	trackbar1(trackbarValue, userDataImg, g_imgCanny);
}
void trackbar1(int value, Mat& userdataimg, Mat& imgdst){
	int apertureSize = g_nKSize * 2 + 1;
	if(apertureSize < 3){
		apertureSize = 3;
	}
	Canny(userdataimg, imgdst, g_nCannyMinThreshold, g_nCannyMaxThreshold, apertureSize);
	imshow("imgCanny", imgdst);
}
void trackbarCallback2(int trackbarValue, void* userdata){
	/* canny max */
	cout << "trackbarCallback2" << endl;
	Mat& userDataImg = *(Mat*)userdata;
	trackbar2(trackbarValue, userDataImg, g_imgCanny);
}
void trackbar2(int value, Mat& userdataimg, Mat& imgdst){
	trackbar1(value, userdataimg, imgdst);
}

/* morphology */
void trackbarCallback3(int trackbarValue, void* userdata){
	/* morphology */
	cout << "trackbarCallback3" << endl;
	Mat& userDataImg = *(Mat*)userdata;
	trackbar3(trackbarValue, userDataImg, g_imgMorph);
}
void trackbar3(int value, Mat& userdataimg, Mat& imgdst){
	cout << "trackbar3" << endl;
	if(value > 6){
		return;
	}
	/**/
	//myMorphologyEx(userdataimg, imgdst, value);
	/**/
}

/* threshold */
void trackbarCallback4(int trackbarValue, void* userdata){
	/* threshold */
	cout << "trackbarCallback4" << endl;
	Mat& userDataImg = *(Mat*)userdata;
	trackbar4(trackbarValue, userDataImg, g_imgThreshod);
}
void trackbar4(int value, Mat& userdataimg, Mat& imgdst){
	cout << "trackbar4" << endl;
	int nBSize = g_nThresholdBSize * 2 + 1;
	if(nBSize < 3){
		userdataimg.copyTo(imgdst);
	}
	else{
		adaptiveThreshold(userdataimg, imgdst, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, nBSize, g_nThresholdMeanC*0.01);
	}
	imshow("imgThreshold", imgdst);
}
void trackbarCallback5(int trackbarValue, void* userdata){
	/* threshold */
	cout << "trackbarCallback5" << endl;
	Mat& userDataImg = *(Mat*)userdata;
	trackbar5(trackbarValue, userDataImg, g_imgMorph);
}
void trackbar5(int value, Mat& userdataimg, Mat& imgdst){
	cout << "trackbar5" << endl;
	trackbar4(value, userdataimg, imgdst);
}

/* good match */
void trackbarCallback6(int trackbarvalue, void* userdata){
	/**
	cout<<"trackbarCallback6"<<endl;
	/**/
	Mat& img = *(Mat*)userdata;
	trackbar6(trackbarvalue, img);
}
void trackbar6(int value, Mat& img){
	/**
	cout << "trackbar6" << endl;
	/**/
}
/**/
void mouseCallback(int event, int x, int y, int flags, void* userdata){
	if(event == EVENT_RBUTTONDOWN){
		go();
	}
}

