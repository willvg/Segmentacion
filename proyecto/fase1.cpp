/*#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <stdlib.h>

#include <string> 
#include <sstream> 
#include <ctime>
#include <chrono>
*/
#include <highgui.h>
#include <opencv2/opencv.hpp>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include <iostream>
#include <stdio.h>
#include <fstream> 

/**
 * Esta funcion se encarga de detectar los puntos de interes en la imagen original y en la mascara
 * @param image      imagen de entrada
 * @param imageMask  mascara
 * @param keypoints1 vector de puntos de la imagen de entrada
 * @param keypoints2 vector de puntos de la mascara
 */
void detectedPoints(
	cv::Mat 		   image, 
	cv::Mat 		   imageMask, 
	std::vector<cv::KeyPoint>& keypoints1, 
	std::vector<cv::KeyPoint>& keypoints2)
{
    cv::SurfFeatureDetector detector(2000);

    //detecta los puntos de La dos imagenes
    detector.detect(image, keypoints1);
    detector.detect(imageMask, keypoints2);

    //objetos para visualizar los puntos en la imagen
    cv::Mat img_keypoints_1;
    cv::drawKeypoints( image, keypoints1, img_keypoints_1, cv::Scalar::all(-1), 0); //DrawMatchesFlags::DEFAULT

    cv::imshow("Puntos de la imagen", img_keypoints_1 );
}

/**
 * este metodo se encarga de la creacion de los descriptores
 * @param image       imagen
 * @param keypoints   vector de puntoa de interes 
 * @param descriptors el descriptore
 */
void createDescriptor(
	cv::Mat 		  image, 
	std::vector<cv::KeyPoint> keypoints,
	cv::Mat& 		  descriptors)
{
    // obtine los descriptores de las dos imagenes
    cv::SurfDescriptorExtractor extractor;
    extractor.compute(image, keypoints, descriptors);
}

/**
 * Crea el FLan para luego ser guardado
 * @param descriptors1 descriptor de la imagen original
 * @param descriptors2 descriptor de la imagen de mascara
 * @param matches      vector de coincidencias
 * @param max_dist     maxima distancia de puntos
 * @param min_dist     minima distacia que hay entre los puntos
 * @param good_matches vector con las mejores coincidencias
 */
void createFLAN(
	cv::Mat 		   descriptors1, 
	cv::Mat 		   descriptors2,
	std::vector< cv::DMatch >& matches, 
        double& 		   max_dist, 
	double& 		   min_dist, 
	std::vector< cv::DMatch >& good_matches)
{

    cv::FlannBasedMatcher matcher;

    //obtine los match
    matcher.match( descriptors1, descriptors2, matches );

    //Calcula la maxima y la minima distacia entre los puntos.
    for( int i = 0; i < descriptors1.rows; i++ )
    { double dist = matches[i].distance;
      if( dist < min_dist ) min_dist = dist;
      if( dist > max_dist ) max_dist = dist;
    }

    //obtengo las coincidencias que realmente realizan los maches cumpliendo entre el rango establecido
    for( int i = 0; i < descriptors1.rows; i++ )
    { if( matches[i].distance <= cv::max(2*min_dist, 0.002) )
      { good_matches.push_back( matches[i]); }
    }
}

/**
 * encuentra el contorno y lo pinta con respecto a la imagen original
 * @param image     imagen original en escala de grises
 * @param contours  vector de contorno
 * @param hierarchy 
 */
void findContoursImage(
	cv::Mat 			      image, 
	std::vector<std::vector<cv::Point> >& contours, 
	std::vector<cv::Vec4i>& 	      hierarchy)
{
    cv::Canny(image, image, 100, 200, 3);
    cv::RNG rng(12345);
    cv::findContours( image, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0) );
}


/**
 * Este metodo es el encargado de leer las direcciones donde estas las mariposas
 * @param path        path donde estan las direcciones
 * @param directiones vector con las direciones
 */
int readPathTrain(
	std::string 		  path, 
	std::vector<std::string>& directiones) 
{
    std::ifstream file;
    std::string line;
    
    file.open(path.c_str(), std::ifstream::in);
    if(!file.is_open()){
	std::cout << "Could not open file of images path" << std::endl;
	return 1;	
    }
    
    //Leer el archivo hasta que termine.
    while (!file.eof()) {
        getline(file, line); //Lee la linea
        directiones.push_back(line); //agrega la linea al vector
    }
    return 0;
    
}

/**
 * Este metodo es el encargado de leer las imagenes
 * @param directiones vector con las direcciones
 * @param images      vector con las imagenes originales
 * @param imagesMask  vector con las imagenes de mascara
 */
void readImage(
	std::vector<std::string> directiones, 
	std::vector<cv::Mat>& 	 images, 
	std::vector<cv::Mat>&    imagesMask)
{    
    bool flag = true;
    for (int i = 0; i < directiones.size()-2; i++)
    {
        if (flag)
        {
            cv::Mat image = cv::imread(directiones[i].c_str(), 1);
            cv::Mat imageMask = cv::imread(directiones[i+1].c_str(), 1);

            if(! image.data )// Check for invalid input
            {
              std::cout <<  "No se puede abrir la imagen o no se encuentra la imagen" << "\n" ;
            }
            images.push_back(image);
            imagesMask.push_back(imageMask);
            flag = false;
        }
        else{
            flag = true;
        }
    }
}

//code based on https://github.com/daviddoria/Examples/blob/master/c%2B%2B/OpenCV/MeanShiftSegmentation/MeanShiftSegmentation.cxx
void floodFillPostprocess( cv::Mat& img, const cv::Scalar& colorDiff=cv::Scalar::all(1) )
{
    CV_Assert( !img.empty() );
    cv::RNG rng = cv::theRNG();
    cv::Mat mask( img.rows+2, img.cols+2, CV_8UC1, cv::Scalar::all(0) );
    for( int y = 0; y < img.rows; y++ )
    {
        for( int x = 0; x < img.cols; x++ )
        {
            if( mask.at<uchar>(y+1, x+1) == 0 )
            {
                cv::Scalar newVal( rng(256), rng(256), rng(256) );
                floodFill( img, mask, cv::Point(x,y), newVal, 0, colorDiff, colorDiff );
            }
        }
    }
}

//code based on https://github.com/daviddoria/Examples/blob/master/c%2B%2B/OpenCV/MeanShiftSegmentation/MeanShiftSegmentation.cxx
void meanShiftSegmentation(
	cv::Mat&		   image,
	cv::Mat&		   segmented_image)
{
	int spatialRad, colorRad, maxPyrLevel;
	spatialRad = 10;
        colorRad = 10;
        maxPyrLevel = 1;

	pyrMeanShiftFiltering(image, segmented_image, spatialRad, colorRad, maxPyrLevel );
	floodFillPostprocess( segmented_image, cv::Scalar::all(2) );
}

void drawResults(
	cv::Mat&		   segmented_image,
	cv::Mat&		   result_image,
	std::vector<cv::KeyPoint>& image_detected_keypoints)
{
	int i;
	for(i=0; i<std::vector.size(); i++){
		cv::KeyPoint keypoint = image_detected_keypoints.at(i);
	}
}


int main(int argc, char** argv){

    std::string path = "../../../Database_of_Monarch_Butterflies/pathTrain.txt";
    std::vector<std::string> directiones;

    //lee el path donde estan las direciones.
    if(readPathTrain(path,directiones)){
	return 0;
    }

    //for (int i = 0; i < directiones.size(); ++i)
    //{
      //std::cout << directiones[i] << "\n";
    //}

    std::vector<cv::Mat> images; 
    std::vector<cv::Mat> imagesMask;

    //lee las imagenes de entrenamiento originales y las mascaras
    readImage(directiones,images,imagesMask);
        
    for (int i = 0; i < images.size(); ++i)
    {
        cv::Mat image = images[i];
	cv::Mat segmented_image;
        cv::Mat imageMask = imagesMask[i];
       
	//aqui se realiza el entrenamiento
	meanShiftSegmentation(image, segmented_image);
	imshow("segmentation", segmented_image);
       
        cv::waitKey(2000);
    }
  return 0;
}
