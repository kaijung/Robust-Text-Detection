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
#define GET_MSER_REGION_DIR_EN


#ifdef GET_MSER_REGION_DIR_EN

//紀錄mser中間處理影像
Mat mser_middle_process_img_1,mser_middle_process_img_2;
int mser_middle_process_img_color=0;
void get_middle_process_img_start(int width,int height)
{
	mser_middle_process_img_1= Mat( Size(width,height), CV_8UC1, Scalar(0));
	mser_middle_process_img_2 = Mat( Size(width,height), CV_8UC1, Scalar(0));	

	for (int y = 0; y < height; y++) {
	    for(int x = 0; x < width; x++) {
			point_gray_pixel(mser_middle_process_img_1,x,y)=0;
			point_gray_pixel(mser_middle_process_img_2,x,y)=0;
	    }
	}	


	
}

//pixel值128代表 darker to brighter (MSER-)
//pixel值255代表 brighter to darker (MSER+)
//如果是0代表沒寫入過，可以直接寫入哪個方向
//如果大於0代表寫入過，給值64代表重疊

void get_middle_process_img_pix(int curr_x,int curr_y)
{
		if(mser_middle_process_img_color<0)	
			point_gray_pixel(mser_middle_process_img_1,curr_x,curr_y)=255;
		else
			point_gray_pixel(mser_middle_process_img_2,curr_x,curr_y)=255;

/*
	if(point_gray_pixel(image,curr_x,curr_y)==0)
	{
		if(mser_middle_process_img_color<0)	
			point_gray_pixel(image,curr_x,curr_y)=128;
		else
			point_gray_pixel(image,curr_x,curr_y)=255;
	}else{
		point_gray_pixel(image,curr_x,curr_y)=64;
	}

	*/
/*
point_gray_pixel(image,curr_x,curr_y)=255;
*/
	
}
void get_middle_img_color(int color)
{
	mser_middle_process_img_color=color;
}
void get_middle_process_img_end(void)
{

	imwrite("zzz_mser_middle_process_img_1.bmp",mser_middle_process_img_1);
	imwrite("zzz_mser_middle_process_img_2.bmp",mser_middle_process_img_2);

	mser_middle_process_img_1.release();
	mser_middle_process_img_2.release();	
}

//利用MSER產生的中間影像來判斷是否該反向
//只寫入只有一邊有的，同時兩邊都有的不作用
int get_mser_middle_process_flag(Mat &image1,Mat &image2,int curr_x,int curr_y)
{
	if(point_gray_pixel(image2,curr_x,curr_y)>0)
		return 1;
	//if((point_gray_pixel(image2,curr_x,curr_y)>0)&&(point_gray_pixel(image1,curr_x,curr_y)>0)) 
	//	return 1;

	return 0;
}


struct MSERParams
{
    MSERParams( int _delta, int _minArea, int _maxArea, double _maxVariation,
                double _minDiversity, int _maxEvolution, double _areaThreshold,
                double _minMargin, int _edgeBlurSize )
        : delta(_delta), minArea(_minArea), maxArea(_maxArea), maxVariation(_maxVariation),
        minDiversity(_minDiversity), maxEvolution(_maxEvolution), areaThreshold(_areaThreshold),
        minMargin(_minMargin), edgeBlurSize(_edgeBlurSize)
    {}
    int delta;
    int minArea;
    int maxArea;
    double maxVariation;
    double minDiversity;
    int maxEvolution;
    double areaThreshold;
    double minMargin;
    int edgeBlurSize;
};

typedef struct LinkedPoint
{
    struct LinkedPoint* prev;
    struct LinkedPoint* next;
    Point pt;
}
LinkedPoint;

// the history of region grown
typedef struct MSERGrowHistory
{
    struct MSERGrowHistory* shortcut;
    struct MSERGrowHistory* child;
    int stable; // when it ever stabled before, record the size
    int val;
    int size;
}
MSERGrowHistory;

typedef struct MSERConnectedComp
{
    LinkedPoint* head;
    LinkedPoint* tail;
    MSERGrowHistory* history;
    unsigned long grey_level;
    int size;
    int dvar; // the derivative of last var
    float var; // the current variation (most time is the variation of one-step back)
}
MSERConnectedComp;

// clear the connected component in stack
static void
initMSERComp( MSERConnectedComp* comp )
{
    comp->size = 0;
    comp->var = 0;
    comp->dvar = 1;
    comp->history = NULL;
}

// to preprocess src image to following format
// 32-bit image
// > 0 is available, < 0 is visited
// 17~19 bits is the direction
// 8~11 bits is the bucket it falls to (for BitScanForward)
// 0~8 bits is the color
static int* preprocessMSER_8UC1( CvMat* img,
            int*** heap_cur,
            CvMat* src,
            CvMat* mask )
{
    int srccpt = src->step-src->cols;
    int cpt_1 = img->cols-src->cols-1;
    int* imgptr = img->data.i;
    int* startptr;

    int level_size[256];
    for ( int i = 0; i < 256; i++ )
        level_size[i] = 0;

    for ( int i = 0; i < src->cols+2; i++ )
    {
        *imgptr = -1;
        imgptr++;
    }
    imgptr += cpt_1-1;
    uchar* srcptr = src->data.ptr;
    if ( mask )
    {
        startptr = 0;
        uchar* maskptr = mask->data.ptr;
        for ( int i = 0; i < src->rows; i++ )
        {
            *imgptr = -1;
            imgptr++;
            for ( int j = 0; j < src->cols; j++ )
            {
                if ( *maskptr )
                {
                    if ( !startptr )
                        startptr = imgptr;
                    *srcptr = 0xff-*srcptr;
                    level_size[*srcptr]++;
                    *imgptr = ((*srcptr>>5)<<8)|(*srcptr);
                } else {
                    *imgptr = -1;
                }
                imgptr++;
                srcptr++;
                maskptr++;
            }
            *imgptr = -1;
            imgptr += cpt_1;
            srcptr += srccpt;
            maskptr += srccpt;
        }
    } else {
        startptr = imgptr+img->cols+1;
        for ( int i = 0; i < src->rows; i++ )
        {
            *imgptr = -1;
            imgptr++;
            for ( int j = 0; j < src->cols; j++ )
            {
                *srcptr = 0xff-*srcptr;
                level_size[*srcptr]++;
                *imgptr = ((*srcptr>>5)<<8)|(*srcptr);
                imgptr++;
                srcptr++;
            }
            *imgptr = -1;
            imgptr += cpt_1;
            srcptr += srccpt;
        }
    }
    for ( int i = 0; i < src->cols+2; i++ )
    {
        *imgptr = -1;
        imgptr++;
    }

    heap_cur[0][0] = 0;
    for ( int i = 1; i < 256; i++ )
    {
        heap_cur[i] = heap_cur[i-1]+level_size[i-1]+1;
        heap_cur[i][0] = 0;
    }
    return startptr;
}

// add a pixel to the pixel list
static void accumulateMSERComp( MSERConnectedComp* comp, LinkedPoint* point )
{
    if ( comp->size > 0 )
    {
        point->prev = comp->tail;
        comp->tail->next = point;
        point->next = NULL;
    } else {
        point->prev = NULL;
        point->next = NULL;
        comp->head = point;
    }
    comp->tail = point;
    comp->size++;
}


static float
MSERVariationCalc( MSERConnectedComp* comp, int delta )
{
    MSERGrowHistory* history = comp->history;
    int val = comp->grey_level;
    if ( NULL != history )
    {
        MSERGrowHistory* shortcut = history->shortcut;
        while ( shortcut != shortcut->shortcut && shortcut->val + delta > val )
            shortcut = shortcut->shortcut;
        MSERGrowHistory* child = shortcut->child;
        while ( child != child->child && child->val + delta <= val )
        {
            shortcut = child;
            child = child->child;
        }
        // get the position of history where the shortcut->val <= delta+val and shortcut->child->val >= delta+val
        history->shortcut = shortcut;
        return (float)(comp->size-shortcut->size)/(float)shortcut->size;
        // here is a small modification of MSER where cal ||R_{i}-R_{i-delta}||/||R_{i-delta}||
        // in standard MSER, cal ||R_{i+delta}-R_{i-delta}||/||R_{i}||
        // my calculation is simpler and much easier to implement
    }
    return 1.;
}

static bool MSERStableCheck( MSERConnectedComp* comp, MSERParams params )
{
    // tricky part: it actually check the stablity of one-step back
    if ( comp->history == NULL || comp->history->size <= params.minArea || comp->history->size >= params.maxArea )
        return 0;
    float div = (float)(comp->history->size-comp->history->stable)/(float)comp->history->size;
    float var = MSERVariationCalc( comp, params.delta );
    int dvar = ( comp->var < var || (unsigned long)(comp->history->val + 1) < comp->grey_level );
    int stable = ( dvar && !comp->dvar && comp->var < params.maxVariation && div > params.minDiversity );
    comp->var = var;
    comp->dvar = dvar;
    if ( stable )
        comp->history->stable = comp->history->size;
    return stable != 0;
}

// convert the point set to CvSeq
static CvContour* MSERToContour( MSERConnectedComp* comp, CvMemStorage* storage )
{
    CvSeq* _contour = cvCreateSeq( CV_SEQ_KIND_GENERIC|CV_32SC2, sizeof(CvContour), sizeof(CvPoint), storage );
    CvContour* contour = (CvContour*)_contour;
    cvSeqPushMulti( _contour, 0, comp->history->size );
    LinkedPoint* lpt = comp->head;
    for ( int i = 0; i < comp->history->size; i++ )
    {
        CvPoint* pt = CV_GET_SEQ_ELEM( CvPoint, _contour, i );
        pt->x = lpt->pt.x;
        pt->y = lpt->pt.y;
        lpt = lpt->next;

		//printf("(%d %d)",lpt->pt.x,lpt->pt.y);
		get_middle_process_img_pix(lpt->pt.x,lpt->pt.y);
    }
    cvBoundingRect( contour );
    return contour;
}


// add history of size to a connected component
static void
MSERNewHistory( MSERConnectedComp* comp, MSERGrowHistory* history )
{
    history->child = history;
    if ( NULL == comp->history )
    {
        history->shortcut = history;
        history->stable = 0;
    } else {
        comp->history->child = history;
        history->shortcut = comp->history->shortcut;
        history->stable = comp->history->stable;
    }
    history->val = comp->grey_level;
    history->size = comp->size;
    comp->history = history;
}

// merging two connected component
static void
MSERMergeComp( MSERConnectedComp* comp1,
          MSERConnectedComp* comp2,
          MSERConnectedComp* comp,
          MSERGrowHistory* history )
{
    LinkedPoint* head;
    LinkedPoint* tail;
    comp->grey_level = comp2->grey_level;
    history->child = history;
    // select the winner by size
    if ( comp1->size >= comp2->size )
    {
        if ( NULL == comp1->history )
        {
            history->shortcut = history;
            history->stable = 0;
        } else {
            comp1->history->child = history;
            history->shortcut = comp1->history->shortcut;
            history->stable = comp1->history->stable;
        }
        if ( NULL != comp2->history && comp2->history->stable > history->stable )
            history->stable = comp2->history->stable;
        history->val = comp1->grey_level;
        history->size = comp1->size;
        // put comp1 to history
        comp->var = comp1->var;
        comp->dvar = comp1->dvar;
        if ( comp1->size > 0 && comp2->size > 0 )
        {
            comp1->tail->next = comp2->head;
            comp2->head->prev = comp1->tail;
        }
        head = ( comp1->size > 0 ) ? comp1->head : comp2->head;
        tail = ( comp2->size > 0 ) ? comp2->tail : comp1->tail;
        // always made the newly added in the last of the pixel list (comp1 ... comp2)
    } else {
        if ( NULL == comp2->history )
        {
            history->shortcut = history;
            history->stable = 0;
        } else {
            comp2->history->child = history;
            history->shortcut = comp2->history->shortcut;
            history->stable = comp2->history->stable;
        }
        if ( NULL != comp1->history && comp1->history->stable > history->stable )
            history->stable = comp1->history->stable;
        history->val = comp2->grey_level;
        history->size = comp2->size;
        // put comp2 to history
        comp->var = comp2->var;
        comp->dvar = comp2->dvar;
        if ( comp1->size > 0 && comp2->size > 0 )
        {
            comp2->tail->next = comp1->head;
            comp1->head->prev = comp2->tail;
        }
        head = ( comp2->size > 0 ) ? comp2->head : comp1->head;
        tail = ( comp1->size > 0 ) ? comp1->tail : comp2->tail;
        // always made the newly added in the last of the pixel list (comp2 ... comp1)
    }
    comp->head = head;
    comp->tail = tail;
    comp->history = history;
    comp->size = comp1->size + comp2->size;
}

static void extractMSER_8UC1_Pass( int* ioptr,
              int* imgptr,
              int*** heap_cur,
              LinkedPoint* ptsptr,
              MSERGrowHistory* histptr,
              MSERConnectedComp* comptr,
              int step,
              int stepmask,
              int stepgap,
              MSERParams params,
              int color,
              CvSeq* contours,
              CvMemStorage* storage )
{
    comptr->grey_level = 256;
    comptr++;
    comptr->grey_level = (*imgptr)&0xff;
    initMSERComp( comptr );
    *imgptr |= 0x80000000;
    heap_cur += (*imgptr)&0xff;
    int dir[] = { 1, step, -1, -step };
#ifdef __INTRIN_ENABLED__
    unsigned long heapbit[] = { 0, 0, 0, 0, 0, 0, 0, 0 };
    unsigned long* bit_cur = heapbit+(((*imgptr)&0x700)>>8);
#endif
    for ( ; ; )
    {
        // take tour of all the 4 directions
        while ( ((*imgptr)&0x70000) < 0x40000 )
        {
            // get the neighbor
            int* imgptr_nbr = imgptr+dir[((*imgptr)&0x70000)>>16];
            if ( *imgptr_nbr >= 0 ) // if the neighbor is not visited yet
            {
                *imgptr_nbr |= 0x80000000; // mark it as visited
                if ( ((*imgptr_nbr)&0xff) < ((*imgptr)&0xff) )
                {
                    // when the value of neighbor smaller than current
                    // push current to boundary heap and make the neighbor to be the current one
                    // create an empty comp
                    (*heap_cur)++;
                    **heap_cur = imgptr;
                    *imgptr += 0x10000;
                    heap_cur += ((*imgptr_nbr)&0xff)-((*imgptr)&0xff);
#ifdef __INTRIN_ENABLED__
                    _bitset( bit_cur, (*imgptr)&0x1f );
                    bit_cur += (((*imgptr_nbr)&0x700)-((*imgptr)&0x700))>>8;
#endif
                    imgptr = imgptr_nbr;
                    comptr++;
                    initMSERComp( comptr );
                    comptr->grey_level = (*imgptr)&0xff;
                    continue;
                } else {
                    // otherwise, push the neighbor to boundary heap
                    heap_cur[((*imgptr_nbr)&0xff)-((*imgptr)&0xff)]++;
                    *heap_cur[((*imgptr_nbr)&0xff)-((*imgptr)&0xff)] = imgptr_nbr;
#ifdef __INTRIN_ENABLED__
                    _bitset( bit_cur+((((*imgptr_nbr)&0x700)-((*imgptr)&0x700))>>8), (*imgptr_nbr)&0x1f );
#endif
                }
            }
            *imgptr += 0x10000;
        }
        int imsk = (int)(imgptr-ioptr);
        ptsptr->pt = cvPoint( imsk&stepmask, imsk>>stepgap );
        // get the current location
        accumulateMSERComp( comptr, ptsptr );
        ptsptr++;
        // get the next pixel from boundary heap
        if ( **heap_cur )
        {
            imgptr = **heap_cur;
            (*heap_cur)--;
#ifdef __INTRIN_ENABLED__
            if ( !**heap_cur )
                _bitreset( bit_cur, (*imgptr)&0x1f );
#endif
        } else {
#ifdef __INTRIN_ENABLED__
            bool found_pixel = 0;
            unsigned long pixel_val;
            for ( int i = ((*imgptr)&0x700)>>8; i < 8; i++ )
            {
                if ( _BitScanForward( &pixel_val, *bit_cur ) )
                {
                    found_pixel = 1;
                    pixel_val += i<<5;
                    heap_cur += pixel_val-((*imgptr)&0xff);
                    break;
                }
                bit_cur++;
            }
            if ( found_pixel )
#else
            heap_cur++;
            unsigned long pixel_val = 0;
            for ( unsigned long i = ((*imgptr)&0xff)+1; i < 256; i++ )
            {
                if ( **heap_cur )
                {
                    pixel_val = i;
                    break;
                }
                heap_cur++;
            }
            if ( pixel_val )
#endif
            {
                imgptr = **heap_cur;
                (*heap_cur)--;
#ifdef __INTRIN_ENABLED__
                if ( !**heap_cur )
                    _bitreset( bit_cur, pixel_val&0x1f );
#endif
                if ( pixel_val < comptr[-1].grey_level )
                {
                    // check the stablity and push a new history, increase the grey level
                    if ( MSERStableCheck( comptr, params ) )
                    {
						//printf("-----------color=%d-----------\n",color);
						get_middle_img_color(color);

						
                        CvContour* contour = MSERToContour( comptr, storage );
                        contour->color = color;
                        cvSeqPush( contours, &contour );
                    }
                    MSERNewHistory( comptr, histptr );
                    comptr[0].grey_level = pixel_val;
                    histptr++;
                } else {
                    // keep merging top two comp in stack until the grey level >= pixel_val
                    for ( ; ; )
                    {
                        comptr--;
                        MSERMergeComp( comptr+1, comptr, comptr, histptr );
                        histptr++;
                        if ( pixel_val <= comptr[0].grey_level )
                            break;
                        if ( pixel_val < comptr[-1].grey_level )
                        {
                            // check the stablity here otherwise it wouldn't be an ER
                            if ( MSERStableCheck( comptr, params ) )
                            {
									//printf("-----------color=%d-----------\n",color);
									get_middle_img_color(color);
														
                                CvContour* contour = MSERToContour( comptr, storage );
                                contour->color = color;
                                cvSeqPush( contours, &contour );
                            }
                            MSERNewHistory( comptr, histptr );
                            comptr[0].grey_level = pixel_val;
                            histptr++;
                            break;
                        }
                    }
                }
            } else
                break;
        }
    }
}


static void extractMSER_8UC1( CvMat* src,
             CvMat* mask,
             CvSeq* contours,
             CvMemStorage* storage,
             MSERParams params )
{
    int step = 8;
    int stepgap = 3;
    while ( step < src->step+2 )
    {
        step <<= 1;
        stepgap++;
    }
    int stepmask = step-1;

    // to speedup the process, make the width to be 2^N
    CvMat* img = cvCreateMat( src->rows+2, step, CV_32SC1 );
    int* ioptr = img->data.i+step+1;
    int* imgptr;

    // pre-allocate boundary heap
    int** heap = (int**)cvAlloc( (src->rows*src->cols+256)*sizeof(heap[0]) );
    int** heap_start[256];
    heap_start[0] = heap;

    // pre-allocate linked point and grow history
    LinkedPoint* pts = (LinkedPoint*)cvAlloc( src->rows*src->cols*sizeof(pts[0]) );
    MSERGrowHistory* history = (MSERGrowHistory*)cvAlloc( src->rows*src->cols*sizeof(history[0]) );
    MSERConnectedComp comp[257];



get_middle_process_img_start(src->cols,src->rows);


    // darker to brighter (MSER-)
    imgptr = preprocessMSER_8UC1( img, heap_start, src, mask );
    extractMSER_8UC1_Pass( ioptr, imgptr, heap_start, pts, history, comp, step, stepmask, stepgap, params, -1, contours, storage );

//get_middle_process_img_end();

//get_middle_process_img_start(src->cols,src->rows);
		
    // brighter to darker (MSER+)
    imgptr = preprocessMSER_8UC1( img, heap_start, src, mask );
    extractMSER_8UC1_Pass( ioptr, imgptr, heap_start, pts, history, comp, step, stepmask, stepgap, params, 1, contours, storage );


//get_middle_process_img_end();

    // clean up
    cvFree( &history );
    cvFree( &heap );
    cvFree( &pts );
    cvReleaseMat( &img );
}



static void
extractMSER( CvArr* _img,
           CvArr* _mask,
           CvSeq** _contours,
           CvMemStorage* storage,
           MSERParams params )
{
    CvMat srchdr, *src = cvGetMat( _img, &srchdr );
    CvMat maskhdr, *mask = _mask ? cvGetMat( _mask, &maskhdr ) : 0;
    CvSeq* contours = 0;

    CV_Assert(src != 0);
    CV_Assert(CV_MAT_TYPE(src->type) == CV_8UC1 || CV_MAT_TYPE(src->type) == CV_8UC3);
    CV_Assert(mask == 0 || (CV_ARE_SIZES_EQ(src, mask) && CV_MAT_TYPE(mask->type) == CV_8UC1));
    CV_Assert(storage != 0);

    contours = *_contours = cvCreateSeq( 0, sizeof(CvSeq), sizeof(CvSeq*), storage );

    // choose different method for different image type
    // for grey image, it is: Linear Time Maximally Stable Extremal Regions
    // for color image, it is: Maximally Stable Colour Regions for Recognition and Matching
    switch ( CV_MAT_TYPE(src->type) )
    {
        case CV_8UC1:
            extractMSER_8UC1( src, mask, contours, storage, params );
            break;
       // case CV_8UC3:
       //     extractMSER_8UC3( src, mask, contours, storage, params );
        //    break;
    }
}

void MSER::operator()( const Mat& image, vector<vector<Point> >& dstcontours, const Mat& mask ) const
{
printf("dddd");
    CvMat _image = image, _mask, *pmask = 0;
    if( mask.data )
        pmask = &(_mask = mask);
    MemStorage storage(cvCreateMemStorage(0));
    Seq<CvSeq*> contours;
		
    extractMSER( &_image, pmask, &contours.seq, storage,
                 MSERParams(delta, minArea, maxArea, maxVariation, minDiversity,
                            maxEvolution, areaThreshold, minMargin, edgeBlurSize));
    SeqIterator<CvSeq*> it = contours.begin();
    size_t i, ncontours = contours.size();
    dstcontours.resize(ncontours);
    for( i = 0; i < ncontours; i++, ++it )
        Seq<Point>(*it).copyTo(dstcontours[i]);
		/**/
}
#endif



RobustTextDetection::RobustTextDetection(string temp_img_directory) {
}

RobustTextDetection::RobustTextDetection(RobustTextParam & param, string temp_img_directory) {
    this->param                 = param;
    this->tempImageDirectory    = temp_img_directory;
}
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
    
#ifndef GET_MSER_REGION_DIR_EN
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


#ifdef GET_MSER_REGION_DIR_EN
    for( int y = 1; y < edges.rows - 1; y++ ) {
        uchar * edge_ptr = edges.ptr<uchar>(y);
        uchar * grad_ptr = grad_dir.ptr<uchar>(y);
        uchar * next_ptr = result.ptr<uchar>(y + 1);
        for( int x = 1; x < edges.cols - 1; x++ ) {
            /* Only consider the contours */
            if( edge_ptr[x] != 0 ) {
                /* .. there should be a better way .... */
				int flag=get_mser_middle_process_flag(mser_middle_process_img_1,mser_middle_process_img_2,x,y);
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
            }
        }

        prev_ptr = curr_ptr;
        curr_ptr = next_ptr;
    }



	get_middle_process_img_end();	
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
