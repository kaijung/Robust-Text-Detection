//
//  main.cpp
//  RobustTextDetection
//
//  Created by Saburo Okita on 05/06/14.
//  Copyright (c) 2014 Saburo Okita. All rights reserved.
//

#include <iostream>
#include <bitset>
#include <iterator> 
#include <opencv2/opencv.hpp>
#include <tesseract/baseapi.h>

#include "RobustTextDetection.h"
#include "ConnectedComponent.h"

using namespace std;
using namespace cv;


int main(int argc, const char * argv[])
{


    //namedWindow( "" );
    //moveWindow("", 0, 0);
    Mat image;

    if(argc == 2) 
        image = imread( argv[1] );
    else
        return -1;

    //printf("Line 34\n");

    /* Quite a handful or params */
    RobustTextParam param;
    param.minMSERArea        = 10;
    param.maxMSERArea        = 2000;
    param.cannyThresh1       = 20;
    param.cannyThresh2       = 100;
    
    param.maxConnCompCount   = 3000;
    param.minConnCompArea    = 75;
    param.maxConnCompArea    = 600;
    
    param.minEccentricity    = 0.1;
    param.maxEccentricity    = 0.995;
    param.minSolidity        = 0.4;
    param.maxStdDevMeanRatio = 0.5;
    
    /* Apply Robust Text Detection */
    /* ... remove this temp output path if you don't want it to write temp image files */

    string temp_output_path = "/tmp";

    RobustTextDetection detector(param, temp_output_path );
    pair<Mat, Rect> result = detector.apply( image );
    //printf("Line 56\n");  
    /* Get the region where the candidate text is */
    Mat stroke_width( result.second.height, result.second.width, CV_8UC1, Scalar(0) );
    Mat(result.first, result.second).copyTo( stroke_width);

    /* Use Tesseract to try to decipher our image */
    tesseract::TessBaseAPI tesseract_api;
    tesseract_api.Init(NULL, "eng"  );
    tesseract_api.SetImage((uchar*) stroke_width.data, stroke_width.cols, stroke_width.rows, 1, stroke_width.cols);
    
    string out = string(tesseract_api.GetUTF8Text());

    cout << out << "\n";


    return 0;
}
