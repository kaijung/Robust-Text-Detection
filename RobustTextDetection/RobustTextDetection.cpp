//
//  RobustTextDetection.cpp
//  RobustTextDetection
//
//  Created by Saburo Okita on 08/06/14.
//  Copyright (c) 2014 Saburo Okita. All rights reserved.
//
#include <bitset>
#include "RobustTextDetection.h"
#include "ConnectedComponent.h"

#include <numeric>

using namespace std;
using namespace cv;

#define point_gray_pixel(image,x,y) image.at<uchar>(y,x)
#define GETFONTDIR_EN
#define GROW_PLUS_EN
#define GROW_FACTOR 5.0

#ifdef GETFONTDIR_EN
	Mat Light_image;
#endif


RobustTextDetection::RobustTextDetection(string temp_img_directory) {
}

RobustTextDetection::RobustTextDetection(RobustTextParam & param, string temp_img_directory) {
    this->param                 = param;
    this->tempImageDirectory    = temp_img_directory;
}



#ifdef GETFONTDIR_EN
//把MSCE找出的連通區域算出每個區域的代表亮度和填進去另一張影像內
Mat  Get_Region_Light_image(vector<vector<Point>> &region_group,Mat &original_img)
{
	int curr_x=0,curr_y=0;
	int height,width;
	height=original_img.rows;
	width=original_img.cols;

	//debug
	//imwrite("zzz_original_img.bmp",original_img);

	
	//Mat m=Mat(Size(width,height),type);
    Mat Region_Light_image( Size(width,height), CV_8UC1, Scalar(0));
	//clear buffer
	for (int y = 0; y < height; y++) 
	{
		for(int x = 0; x <width; x++) 
		{
			Region_Light_image.at<uchar>(y,x)=255;
		}
	}	


	
	int histogram[256]={0};
	uchar curr_gray_level=0;
	int aa=region_group.size();

	
	for(int i= 0; i < region_group.size(); i++)
	{
		//clear buffer
		for(int ii= 0; ii < 256; ii++)
			histogram[ii]=0;

		int bb=region_group[i].size();





#if 0		
		//統計直方圖	
		for(int j= 0; j < region_group[i].size();j++) 
		{
			curr_x=region_group[i][j].x;
			curr_y=region_group[i][j].y;
			curr_gray_level=original_img.at<uchar>(curr_y,curr_x);
			histogram[curr_gray_level]++;
		}
		//找出票數最多的灰階值
		uchar gray_level=find_max_return_index(histogram,256);
#else
		//計算平均值
		long sum=0;
		int count=0;
		for(int j= 0; j < region_group[i].size();j++) 
		{
			curr_x=region_group[i][j].x;
			curr_y=region_group[i][j].y;
			curr_gray_level=original_img.at<uchar>(curr_y,curr_x);
			sum+=curr_gray_level;
			count++;
		}
		//找出票數最多的灰階值
		uchar gray_level=sum/count;		
#endif
		//把原本該區域都填成該灰階值
		//if(i!=30)
			//continue;



		for(int j= 0; j < region_group[i].size();j++) 
		{
			curr_x=region_group[i][j].x;
			curr_y=region_group[i][j].y;
			curr_gray_level=Region_Light_image.at<uchar>(curr_y,curr_x);
			//if(curr_gray_level==255)
			Region_Light_image.at<uchar>(curr_y,curr_x)=gray_level;
		}


			//char StringTmp[1024];
			//sprintf(StringTmp,"zzz_Region_Light_image_%d.bmp",i);
			//imwrite(StringTmp,Region_Light_image);
	}


	return Region_Light_image;
}


//比較兩點的距離，倒底是距離哪點比較遠
int Get_Dir(uchar font_level,uchar level_1,uchar level_2)
{
	int diff1=abs(font_level-level_1);
	int diff2=abs(font_level-level_2);	
	if(diff1>diff2)
		return 0;
	else
		return 1;
}
int bound(int value,int lowbound,int upbound)
{
		if(value>=upbound) 
			return upbound;
		if(value<=lowbound) 
			return lowbound;
		return value;
}
//給方向、起始點和強度開始做區域擴散
//input dir edge_level start_xy image
//dir_flag_inv代表顛倒原來方向
/* Convert the angle into predefined 3x3 neighbor locations
	| 2 | 3 | 4 |
	| 1 | 0 | 5 |
	| 8 | 7 | 6 |
*/
void Grow_dir(int dir,int dir_flag_inv,float curr_edge_level,Mat &image,int start_x,int start_y)
{
		
		int edge_level=curr_edge_level+0.5;

//printf("%d ",edge_level);
		//edge_level=bound(edge_level,0,3);
		int curr_x=0,curr_y=0;
		int true_dir=0;//根據flag改成正確方向
		true_dir=dir;

		if(dir_flag_inv)
		{
			if(dir==1)
			{
				true_dir=5;
			}else if(dir==2)
			{
				true_dir=6;
			}else if(dir==3)
			{
				true_dir=7;
			}else if(dir==4)
			{
				true_dir=8;
			}else if(dir==5)
			{
				true_dir=1;
			}else if(dir==6)
			{
				true_dir=2;
			}else if(dir==7)
			{
				true_dir=3;
			}else if(dir==8)
			{
				true_dir=4;
			}
		}



	    /* 
	     | 2 | 3 | 4 |
	     | 1 | 0 | 5 |
	     | 8 | 7 | 6 |
	     */
		int loop_count=0;		
		while(1)
		{
			if(loop_count>=edge_level)
				return;
			
			if(true_dir==1)
			{

				curr_x=start_x-(loop_count+1);
				curr_y=start_y;			
			}else if(true_dir==2)
			{
				curr_x=start_x-(loop_count+1);
				curr_y=start_y-(loop_count+1);	
			}else if(true_dir==3)
			{
				curr_x=start_x;
				curr_y=start_y-(loop_count+1);	
			}else if(true_dir==4)
			{
				curr_x=start_x+(loop_count+1);
				curr_y=start_y-(loop_count+1);	
			}else if(true_dir==5)
			{
				curr_x=start_x+(loop_count+1);
				curr_y=start_y;		
			}else if(true_dir==6)
			{
				curr_x=start_x+(loop_count+1);
				curr_y=start_y+(loop_count+1);	
			}else if(true_dir==7)
			{
				curr_x=start_x;
				curr_y=start_y+(loop_count+1);	
			}else if(true_dir==8)
			{
				curr_x=start_x-(loop_count+1);
				curr_y=start_y+(loop_count+1);	
			}

			if(curr_x<0)
				break;
			else if(curr_y<0)
				break;
			else if(curr_x>=image.cols)
				break;
			else if(curr_y>=image.rows)
				break;		

			//image.at<uchar>(curr_y,curr_x)=255;
			point_gray_pixel(image,curr_x,curr_y)=255;

			
			loop_count++;
		}

		

}

#endif




/**
 * Apply robust text detection algorithm
 * It returns the filtered stroke width image which contains the possible
 * text in binary format, and also the rect
 **/
pair<Mat, Rect> RobustTextDetection::apply( Mat& image ) {
    Mat grey      = preprocessImage( image );
    Mat mser_mask = createMSERMask( grey );
    
    
    /* Perform canny edge operator to extract the edges */
    Mat edges;
    Canny( grey, edges, param.cannyThresh1, param.cannyThresh2 );
    
    
    /* Create the edge enhanced MSER region */
    Mat edge_mser_intersection  = edges & mser_mask;
    Mat gradient_grown          = growEdges( grey, edge_mser_intersection );
    Mat edge_enhanced_mser      = ~gradient_grown & mser_mask;
    
    /* Writing temporary output images */
    if( !tempImageDirectory.empty() ) {
        //cout << "Writing temp output images" << endl;
        imwrite( tempImageDirectory + "/out_grey.png",                   grey );
        imwrite( tempImageDirectory + "/out_mser_mask.png",              mser_mask );
        imwrite( tempImageDirectory + "/out_canny_edges.png",            edges );
        imwrite( tempImageDirectory + "/out_edge_mser_intersection.png", edge_mser_intersection );
        imwrite( tempImageDirectory + "/out_gradient_grown.png",         gradient_grown );
        imwrite( tempImageDirectory + "/out_edge_enhanced_mser.png",     edge_enhanced_mser );
    }

#if 1
        imwrite("out_grey.png",                   grey );
        imwrite("out_mser_mask.png",              mser_mask );
        imwrite("out_canny_edges.png",            edges );
        imwrite("out_edge_mser_intersection.png", edge_mser_intersection );
        imwrite("out_gradient_grown.png",         gradient_grown );
        imwrite("out_edge_enhanced_mser.png",     edge_enhanced_mser );
#endif






    /* Find the connected components */
    ConnectedComponent conn_comp( param.maxConnCompCount, 4);
    Mat labels = conn_comp.apply( edge_enhanced_mser );
    vector<ComponentProperty> props = conn_comp.getComponentsProperties();
    
    
    Mat result( labels.size(), CV_8UC1, Scalar(0));
    for( ComponentProperty& prop: props ) {
        /* Filtered out connected components that aren't within the criteria */
        if( prop.area < param.minConnCompArea || prop.area > param.maxConnCompArea )
            continue;
        
        if( prop.eccentricity < param.minEccentricity || prop.eccentricity > param.maxEccentricity )
            continue;
        
        if( prop.solidity < param.minSolidity )
            continue;
        
        result |= (labels == prop.labelID);
    }
    

    /* Calculate the distance transformed from the connected components */
    cv::distanceTransform( result, result, CV_DIST_L2, 3 );
    result.convertTo( result, CV_32SC1 );
    
    /* Find the stroke width image from the distance transformed */
    Mat stroke_width = computeStrokeWidth( result );
    
    /* Filter the stroke width using connected component again */
    conn_comp   = ConnectedComponent( param.maxConnCompCount, 4);
    labels      = conn_comp.apply( stroke_width );
    props       = conn_comp.getComponentsProperties();
    
    Mat filtered_stroke_width( stroke_width.size(), CV_8UC1, Scalar(0) );
    
    for( ComponentProperty& prop: props ) {
        Mat mask = labels == prop.labelID;
        Mat temp;
        stroke_width.copyTo( temp, mask );
        
        int area = countNonZero( temp );
        
        /* Since we only want to consider the connected component, ignore the zero pixels */
        vector<int> vec = Mat( temp.reshape( 1, temp.rows * temp.cols ) );
        vector<int> nonzero_vec;
        copy_if( vec.begin(), vec.end(), back_inserter(nonzero_vec), [&](int val){
            return val > 0;
        });
        
        /* Find mean and std deviation for the connected components */
        double mean = std::accumulate( nonzero_vec.begin(), nonzero_vec.end(), 0.0 ) / area;
        
        double accum = 0.0;
        for( int val: nonzero_vec )
            accum += (val - mean) * (val - mean );
        double std_dev = sqrt( accum / area );
        
        /* Filter out those which are out of the prespecified ratio */
        if( (std_dev / mean) > param.maxStdDevMeanRatio  )
            continue;
        
        /* Collect the filtered stroke width */
        filtered_stroke_width |= mask;
    }

    /* Use morphological close and open to create a large connected bounding region from the filtered stroke width */
    Mat bounding_region;
    morphologyEx( filtered_stroke_width, bounding_region, MORPH_CLOSE, getStructuringElement( MORPH_ELLIPSE, Size(25, 25)) );
    morphologyEx( bounding_region, bounding_region, MORPH_OPEN, getStructuringElement( MORPH_ELLIPSE, Size(7, 7)) );
    
    /* ... so that we can get an overall bounding rect */
    Mat bounding_region_coord;
    findNonZero( bounding_region, bounding_region_coord );
    Rect bounding_rect = boundingRect( bounding_region_coord );
    
    Mat bounding_mask( filtered_stroke_width.size(), CV_8UC1, Scalar(0) );
    Mat( bounding_mask, bounding_rect ) = 255;
    
    /* Well, add some margin to the bounding rect */
    bounding_rect = Rect( bounding_rect.tl() - Point(5, 5), bounding_rect.br() + Point(5, 5) );
    bounding_rect = clamp( bounding_rect, image.size() );
    
    
    /* Well, discard everything outside of the bounding rectangle */
    filtered_stroke_width.copyTo( filtered_stroke_width, bounding_mask );
    
    return pair<Mat, Rect>( filtered_stroke_width, bounding_rect );
}


Rect RobustTextDetection::clamp( Rect& rect, Size size ) {
    Rect result = rect;
    
    if( result.x < 0 )
        result.x = 0;
    
    if( result.x + result.width > size.width )
        result.width = size.width - result.x;
    
    if( result.y < 0 )
        result.y = 0;
    
    if( result.y + result.height > size.height )
        result.height = size.height - result.y;
    
    return result;
}



/**
 * Create a mask out from the MSER components
 */
Mat RobustTextDetection::createMSERMask( Mat& grey ) {
    /* Find MSER components */
    vector<vector<Point>> contours;
    MSER mser( 8, param.minMSERArea, param.maxMSERArea, 0.25, 0.1, 100, 1.01, 0.03, 5 );
    mser(grey, contours);
    
#ifdef GETFONTDIR_EN
	Light_image=Get_Region_Light_image(contours,grey);
	imwrite("zzz_Light_image.bmp",Light_image);
#endif
		
    /* Create a binary mask out of the MSER */
    Mat mser_mask( grey.size(), CV_8UC1, Scalar(0));
    
    for( int i = 0; i < contours.size(); i++ ) {
        for( Point& point: contours[i] )
            mser_mask.at<uchar>(point) = 255;
    }
    
    return mser_mask;
}


/**
 * Preprocess image
 */
Mat RobustTextDetection::preprocessImage( Mat& image ) {
    /* TODO: Should do contrast enhancement here  */
    Mat grey;
    cvtColor( image, grey, CV_BGR2GRAY );
    return grey;
}

/**
 * From the angle convert into our neighborhood encoding
 * which has the following scheme
 * | 2 | 3 | 4 |
 * | 1 | 0 | 5 |
 * | 8 | 7 | 6 |
 */
int RobustTextDetection::toBin( const float angle, const int neighbors ) {
    const float divisor = 180.0 / neighbors;
    return static_cast<int>( (( floor(angle / divisor)  - 1) / 2) + 1 ) % neighbors + 1;
}

/**
 * Grow the edges along with directon of gradient
 */
Mat RobustTextDetection::growEdges(Mat& image, Mat& edges ) {
    CV_Assert( edges.type() == CV_8UC1 );
 
   Mat delta = image.clone();
/*
    for( int y = 0; y < delta.rows; y++ ) {
        uchar * img_ptr = delta.ptr<uchar>(y);
        for( int x = 0; x < delta.cols; x++ ) {
            img_ptr[x] = abs(img_ptr[x]-128);
        }
    }

    imwrite( tempImageDirectory + "/out_grey_delta.png",                   delta );
*/
    Mat grad_x, grad_y;

    Sobel( image, grad_x, CV_32FC1, 1, 0 );
    Sobel( image, grad_y, CV_32FC1, 0, 1 );

#if 0
    double minval,maxval;
   minMaxLoc(grad_x,&minval,&maxval);
	 double abs_minval=abs(minval);

	 double max=0;
	if(maxval>abs_minval)
		grad_x=grad_x/maxval*255;
	else
		grad_x=grad_x/abs_minval*255;

   minMaxLoc(grad_y,&minval,&maxval);
	 abs_minval=abs(minval);

	if(maxval>abs_minval)
		grad_y=grad_y/maxval*255;
	else
		grad_y=grad_y/abs_minval*255;
#endif

	
    Mat grad_mag, grad_dir;
    cartToPolar( grad_x, grad_y, grad_mag, grad_dir, true );
    
    
    /* Convert the angle into predefined 3x3 neighbor locations
     | 2 | 3 | 4 |
     | 1 | 0 | 5 |
     | 8 | 7 | 6 |
     */
    for( int y = 0; y < grad_dir.rows; y++ ) {
        float * grad_ptr = grad_dir.ptr<float>(y);
        
        for( int x = 0; x < grad_dir.cols; x++ ) {
            if( grad_ptr[x] != 0 )
                grad_ptr[x] = toBin( grad_ptr[x] );
        }
    }
    grad_dir.convertTo( grad_dir, CV_8UC1 );
    
    
    
    /* Perform region growing based on the gradient direction */
    Mat result = edges.clone();
    
    uchar * prev_ptr = result.ptr<uchar>(0);
    uchar * curr_ptr = result.ptr<uchar>(1);
    
#ifndef GETFONTDIR_EN	
    for( int y = 1; y < edges.rows - 1; y++ ) {
        uchar * edge_ptr = edges.ptr<uchar>(y);
        uchar * grad_ptr = grad_dir.ptr<uchar>(y);
        uchar * next_ptr = result.ptr<uchar>(y + 1);
        
        for( int x = 1; x < edges.cols - 1; x++ ) {
            /* Only consider the contours */
            if( edge_ptr[x] != 0 ) {
                
                /* .. there should be a better way .... */
                switch( grad_ptr[x] ) {			
                    case 1: curr_ptr[x-1] = 255; break;
                    case 2: prev_ptr[x-1] = 255; break;
                    case 3: prev_ptr[x  ] = 255; break;
                    case 4: prev_ptr[x+1] = 255; break;
                    case 5: curr_ptr[x  ] = 255; break;
                    case 6: next_ptr[x+1] = 255; break;
                    case 7: next_ptr[x  ] = 255; break;
                    case 8: next_ptr[x-1] = 255; break;
/*
                    case 1: curr_ptr[x+1] = 255; break;
                    case 2: next_ptr[x+1] = 255; break;
                    case 3: next_ptr[x  ] = 255; break;
                    case 4: next_ptr[x-1] = 255; break;
                    case 5: curr_ptr[x-1] = 255; break;
                    case 6: prev_ptr[x-1] = 255; break;
                    case 7: prev_ptr[x  ] = 255; break;
                    case 8: prev_ptr[x+1] = 255; break;	
*/										
                    default: break;										
                }
            }
        }
        
        prev_ptr = curr_ptr;
        curr_ptr = next_ptr;
    }
    
#endif



#ifdef GETFONTDIR_EN


	#ifdef GROW_PLUS_EN
		double grad_mag_minval,grad_mag_maxval;
		minMaxLoc(grad_mag,&grad_mag_minval,&grad_mag_maxval);

	#endif	 


//float curr_point_edgevalue=grad_mag.at<float>(y,x); 
//printf("%5.0f ",curr_point_edgevalue);



	uchar level_1=0,level_2=0;
	int flag=0;
    for( int y = 1; y < edges.rows - 1; y++ ) 
	{
        uchar * edge_ptr = edges.ptr<uchar>(y);
        uchar * grad_ptr = grad_dir.ptr<uchar>(y);
        uchar * next_ptr = result.ptr<uchar>(y + 1);		
        for( int x = 1; x < edges.cols - 1; x++ ) 
		{	

			if( edge_ptr[x] != 0 )
			{
					if(grad_ptr[x]==1)
					{
						level_1=image.at<uchar>(y,x-1);
						level_2=image.at<uchar>(y,x+1);	
					}else if(grad_ptr[x]==2)
					{
						level_1=image.at<uchar>(y-1,x-1);
						level_2=image.at<uchar>(y+1,x+1);	
					}else if(grad_ptr[x]==3)
					{
						level_1=image.at<uchar>(y-1,x);
						level_2=image.at<uchar>(y+1,x);	
					}else if(grad_ptr[x]==4)
					{
						level_1=image.at<uchar>(y-1,x+1);
						level_2=image.at<uchar>(y+1,x-1);
					}else if(grad_ptr[x]==5)
					{
						level_2=image.at<uchar>(y,x-1);
						level_1=image.at<uchar>(y,x+1);
					}else if(grad_ptr[x]==6)
					{
						level_2=image.at<uchar>(y-1,x-1);
						level_1=image.at<uchar>(y+1,x+1);
					}else if(grad_ptr[x]==7)
					{
						level_2=image.at<uchar>(y-1,x);
						level_1=image.at<uchar>(y+1,x);	
					}else if(grad_ptr[x]==8)
					{
						level_2=image.at<uchar>(y-1,x+1);
						level_1=image.at<uchar>(y+1,x-1);
					}else{
						level_1=0,level_2=0;
					} 
					uchar curr_point_font_grey_level=Light_image.at<uchar>(y,x); 
					flag=Get_Dir(curr_point_font_grey_level,level_1,level_2);

	#ifndef GROW_PLUS_EN							
					if(flag==0)
					{
			                switch( grad_ptr[x] ) {									
			                    case 1: curr_ptr[x-1] = 255; break; 
			                    case 2: prev_ptr[x-1] = 255; break;
			                    case 3: prev_ptr[x  ] = 255; break;
			                    case 4: prev_ptr[x+1] = 255; break;
			                    case 5: curr_ptr[x  ] = 255; break;
			                    case 6: next_ptr[x+1] = 255; break;
			                    case 7: next_ptr[x  ] = 255; break;
			                    case 8: next_ptr[x-1] = 255; break;									
			                    default: break;
			                }
					}else{
			                switch( grad_ptr[x] ) {
			                    case 1: curr_ptr[x+1] = 255; break;
			                    case 2: next_ptr[x+1] = 255; break;
			                    case 3: next_ptr[x  ] = 255; break;
			                    case 4: next_ptr[x-1] = 255; break;
			                    case 5: curr_ptr[x-1] = 255; break;
			                    case 6: prev_ptr[x-1] = 255; break;
			                    case 7: prev_ptr[x  ] = 255; break;
			                    case 8: prev_ptr[x+1] = 255; break;												
			                    default: break;
			                }
					}
	#else	
					float curr_edge_level=grad_mag.at<uchar>(y,x)*GROW_FACTOR/grad_mag_maxval;
					Grow_dir(grad_ptr[x],flag,curr_edge_level,result,x,y);
	#endif




			}
		







		}
	}
	
			
#endif







		
    return result;
}


/**
 * Convert from our encoded 8 bit uchar to the (8) neighboring coordinates
 */
vector<Point> RobustTextDetection::convertToCoords( int x, int y, bitset<8> neighbors ) {
    vector<Point> coords;
    
    if( neighbors[0] ) coords.push_back( Point(x - 1, y    ) );
    if( neighbors[1] ) coords.push_back( Point(x - 1, y - 1) );
    if( neighbors[2] ) coords.push_back( Point(x    , y - 1) );
    if( neighbors[3] ) coords.push_back( Point(x + 1, y - 1) );
    if( neighbors[4] ) coords.push_back( Point(x + 1, y    ) );
    if( neighbors[5] ) coords.push_back( Point(x + 1, y + 1) );
    if( neighbors[6] ) coords.push_back( Point(x    , y + 1) );
    if( neighbors[7] ) coords.push_back( Point(x - 1, y + 1) );
    
    return coords;
}

/**
 * Overloaded function for convertToCoords
 */
vector<Point> RobustTextDetection::convertToCoords( Point& coord, bitset<8> neighbors ) {
    return convertToCoords( coord.x, coord.y, neighbors );
}

/**
 * Overloaded function for convertToCoords
 */
vector<Point> RobustTextDetection::convertToCoords( Point& coord, uchar neighbors ) {
    return convertToCoords( coord.x, coord.y, bitset<8>(neighbors) );
}

/**
 * Get a set of 8 neighbors that are less than given value
 * | 2 | 3 | 4 |
 * | 1 | 0 | 5 |
 * | 8 | 7 | 6 |
 */
inline bitset<8> RobustTextDetection::getNeighborsLessThan( int * curr_ptr, int x, int * prev_ptr, int * next_ptr ) {
    bitset<8> neighbors;
    neighbors[0] = curr_ptr[x-1] == 0 ? 0 : curr_ptr[x-1] < curr_ptr[x];
    neighbors[1] = prev_ptr[x-1] == 0 ? 0 : prev_ptr[x-1] < curr_ptr[x];
    neighbors[2] = prev_ptr[x  ] == 0 ? 0 : prev_ptr[x]   < curr_ptr[x];
    neighbors[3] = prev_ptr[x+1] == 0 ? 0 : prev_ptr[x+1] < curr_ptr[x];
    neighbors[4] = curr_ptr[x+1] == 0 ? 0 : curr_ptr[x+1] < curr_ptr[x];
    neighbors[5] = next_ptr[x+1] == 0 ? 0 : next_ptr[x+1] < curr_ptr[x];
    neighbors[6] = next_ptr[x  ] == 0 ? 0 : next_ptr[x]   < curr_ptr[x];
    neighbors[7] = next_ptr[x-1] == 0 ? 0 : next_ptr[x-1] < curr_ptr[x];
    return neighbors;
}



/**
 * Compute the stroke width image out from the distance transform matrix
 * It will propagate the max values of each connected component from the ridge
 * to outer boundaries
 **/
Mat RobustTextDetection::computeStrokeWidth( Mat& dist ) {
    /* Pad the distance transformed matrix to avoid boundary checking */
    Mat padded( dist.rows + 1, dist.cols + 1, dist.type(), Scalar(0) );
    dist.copyTo( Mat( padded, Rect(1, 1, dist.cols, dist.rows ) ) );
    
    Mat lookup( padded.size(), CV_8UC1, Scalar(0) );
    int * prev_ptr = padded.ptr<int>(0);
    int * curr_ptr = padded.ptr<int>(1);
    
    for( int y = 1; y < padded.rows - 1; y++ ) {
        uchar * lookup_ptr  = lookup.ptr<uchar>(y);
        int * next_ptr      = padded.ptr<int>(y+1);
        
        for( int x = 1; x < padded.cols - 1; x++ ) {
            /* Extract all the neighbors whose value < curr_ptr[x], encoded in 8-bit uchar */
            if( curr_ptr[x] != 0 )
                lookup_ptr[x] = static_cast<uchar>( getNeighborsLessThan(curr_ptr, x, prev_ptr, next_ptr).to_ullong() );
        }
        prev_ptr = curr_ptr;
        curr_ptr = next_ptr;
    }
    
    
    /* Get max stroke from the distance transformed */
    double max_val_double;
    minMaxLoc( padded, 0, &max_val_double );
    int max_stroke = static_cast<int>(round( max_val_double ));
    
    
    for( int stroke = max_stroke; stroke > 0; stroke-- ) {
        Mat stroke_indices_mat;
        findNonZero( padded == stroke, stroke_indices_mat );
        
        vector<Point> stroke_indices;
        stroke_indices_mat.copyTo( stroke_indices );
        
        vector<Point> neighbors;
        for( Point& stroke_index : stroke_indices ) {
            vector<Point> temp = convertToCoords( stroke_index, lookup.at<uchar>(stroke_index) );
            neighbors.insert( neighbors.end(), temp.begin(), temp.end() );
        }
        
        while( !neighbors.empty() ){
            for( Point& neighbor: neighbors )
                padded.at<int>(neighbor) = stroke;
            
            neighbors.clear();
            
            vector<Point> temp( neighbors );
            neighbors.clear();
            
            /* Recursively gets neighbors of the current neighbors */
            for( Point& neighbor: temp ) {
                vector<Point> temp = convertToCoords( neighbor, lookup.at<uchar>(neighbor) );
                neighbors.insert( neighbors.end(), temp.begin(), temp.end() );
            }
        }
    }
    
    return Mat( padded, Rect(1, 1, dist.cols, dist.rows) );
}
