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

void printResults(
	cv::Mat&		   image,
	std::vector<cv::KeyPoint>& image_detected_keypoints)
{
	int i;
	int contIn=0;
	int contOut=0;
	for (i=0; i<vector.size(); i++){
		cv::KeyPoint keyPoint = vector.at(i);
		int pixel = image.at<uchar>(keypoint.pt().y, keypoint.pt().x);
		if (pixel>0){
			contIn++;
		}
		else{
			contOut++;
		}
	}
	std::cout << "Asserted points: " << contIn << std::endl;  
	std::cout << "Wrong points: " << contOut << std::endl;  
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

    //lee las imagenes de originales y las mascaras
    readImage(directiones,images,imagesMask);
        
    for (int i = 0; i < images.size(); ++i)
    {
        cv::Mat image = images[i];
        cv::Mat imageMask = imagesMask[i];
        //************************************************
        //************************************************
        //*************************************************
        //Detectar los puntos
        //Vector para los puntos de la imagen original y la mascara
        /*std::vector<cv::KeyPoint> keypoints1, keypoints2; coordenada x e y de los puntos mariposa
        //llamar a la funcion de detectar los puntos
        detectedPoints(image,imageMask, keypoints1, keypoints2);
        
        
        //************************************************
        //************************************************
        //*************************************************
        //Descriptores
        //llamar a la funcion de descriptores.
        cv::Mat descriptors1, descriptors2;
        createDescriptor(image, keypoints1, descriptors1);
        createDescriptor(imageMask, keypoints2, descriptors2);
        

        //************************************************
        //************************************************
        //*************************************************
        //FLAN y descriptor
        //vector con las coincidencias 
        std::vector< cv::DMatch > matches;
        double max_dist = 0; double min_dist = 100;
        
        //vector con los maches que realmente conincidieron
        std::vector< cv::DMatch > good_matches;
        createFLAN(descriptors1, descriptors2, matches,max_dist, min_dist, good_matches);

        printf("-- Maxima distancia : %f \n", max_dist );
        printf("-- Minima dist : %f \n", min_dist );

        //dibuja las coincidencias
        cv::Mat img_matches;
        cv::drawMatches( image, keypoints1, imageMask, keypoints2,
                     good_matches, img_matches, cv::Scalar::all(-1), cv::Scalar::all(-1),
                     std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

        //mustra las conincidencias
        cv::imshow( "Coincidencias que cumplen", img_matches );

        for( int i = 0; i < (int)good_matches.size(); i++ )
        { printf( "-- G [%d] Puntos de la imagen: %d  -- Puntos de la mascara: %d -- Distancia: %f \n", 
          i, good_matches[i].queryIdx, good_matches[i].trainIdx, good_matches[i].distance); }
        
        //************************************************
        //************************************************
        //*************************************************
        //Para encontrar los contornos
        cv::Mat gray=image;
        // vector de los contornos   
        std::vector<std::vector<cv::Point> > contours;
            std::vector<cv::Vec4i> hierarchy;
        findContoursImage(image, contours, hierarchy);
        std::cout <<  "---------------------------------------------------------------------------------------------" << "\n" ;
        
        
        /// pintar el contorno
        cv::Mat drawing = cv::Mat::zeros( gray.size(), CV_8UC3 );
        for( int i = 0; i< contours.size(); i++ )
        {
            cv::Scalar color = cv::Scalar( 255, 255, 255 );



            cv::drawContours( drawing, contours, i, color, 1, 8, hierarchy, 0, cv::Point() );
        }     
     
        cv::imshow( "Resultado del contorno", drawing );
        
        std::cout <<  descriptors1.rows << " " << descriptors1.cols<<"\n" ;
        */
       

        //************************************************
        //************************************************
        //*************************************************
        //Para la segmentacion de partes
        
       cv::Mat src = imageMask;

        // Show output image
        imshow("Black Background Image", src);
        // Create a kernel that we will use for accuting/sharpening our image
        cv::Mat kernel = (cv::Mat_<float>(3,3) <<
                1,  1, 1,
                1, -8, 1,
                1,  1, 1); // an approximation of second derivative, a quite strong kernel
        // do the laplacian filtering as it is
        // well, we need to convert everything in something more deeper then CV_8U
        // because the kernel has some negative values,
        // and we can expect in general to have a Laplacian image with negative values
        // BUT a 8bits unsigned int (the one we are working with) can contain values from 0 to 255
        // so the possible negative number will be truncated
        cv::Mat imgLaplacian;
        cv::Mat sharp = src; // copy source image to another temporary one
        cv::filter2D(sharp, imgLaplacian, CV_32F, kernel);
        src.convertTo(sharp, CV_32F);
        cv::Mat imgResult = sharp - imgLaplacian;
	
        // convert back to 8bits gray scale
        imgResult.convertTo(imgResult, CV_8UC3);
        imgLaplacian.convertTo(imgLaplacian, CV_8UC3);
        // imshow( "Laplace Filtered Image", imgLaplacian );
        cv::imshow( "New Sharped Image", imgResult );
        src = imgResult; // copy back
        // Create binary image from source image
        cv::Mat bw;
        cv::cvtColor(src, bw, CV_BGR2GRAY);
        cv::threshold(bw, bw, 40, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
        cv::imshow("Binary Image", bw);

        // Perform the distance transform algorithm
	
	//*****NO FUNCIONA DESDE AQUI
        cv::Mat dist;
        cv::distanceTransform(bw, dist, CV_DIST_L2, 3);
        // Normalize the distance image for range = {0.0, 1.0}
        // so we can visualize and threshold it
        cv::normalize(dist, dist, 0, 1., cv::NORM_MINMAX);
        cv::imshow("Distance Transform Image", dist);
        // Threshold to obtain the peaks
        // This will be the markers for the foreground objects
        cv::threshold(dist, dist, .4, 1., CV_THRESH_BINARY);
        // Dilate a bit the dist image
        cv::Mat kernel1 = cv::Mat::ones(3, 3, CV_8UC1);
        cv::dilate(dist, dist, kernel1);
        cv::imshow("Peaks", dist);
        // Create the CV_8U version of the distance image
        // It is needed for findContours()
        cv::Mat dist_8u;
        dist.convertTo(dist_8u, CV_8U);
        // Find total markers
        std::vector<std::vector<cv::Point> > contours;
        cv::findContours(dist_8u, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
        // Create the marker image for the watershed algorithm
        cv::Mat markers = cv::Mat::zeros(dist.size(), CV_32SC1);
        // Draw the foreground markers
        for (size_t i = 0; i < contours.size(); i++)
            cv::drawContours(markers, contours, static_cast<int>(i), cv::Scalar::all(static_cast<int>(i)+1), -1);
        // Draw the background marker
        cv::circle(markers, cv::Point(5,5), 3, CV_RGB(255,255,255), -1);
        //cv::imshow("Markers", markers*10000);
        // Perform the watershed algorithm
        cv::watershed(src, markers);
        cv::Mat mark = cv::Mat::zeros(markers.size(), CV_8UC1);
        markers.convertTo(mark, CV_8UC1);
        cv::bitwise_not(mark, mark);
    //    imshow("Markers_v2", mark); // uncomment this if you want to see how the mark
                                      // image looks like at that point
        // Generate random colors
        std::vector<cv::Vec3b> colors;
        for (size_t i = 0; i < contours.size(); i++)
        {
            int b = cv::theRNG().uniform(0, 255);
            int g = cv::theRNG().uniform(0, 255);
            int r = cv::theRNG().uniform(0, 255);
            colors.push_back(cv::Vec3b((uchar)b, (uchar)g, (uchar)r));
        }
        // Create the result image
        cv::Mat dst = cv::Mat::zeros(markers.size(), CV_8UC3);

        // Fill labeled objects with random colors
        for (int i = 0; i < markers.rows; i++)
        {
            for (int j = 0; j < markers.cols; j++)
            {
                int index = markers.at<int>(i,j);
                if (index > 0 && index <= static_cast<int>(contours.size()))
                    dst.at<cv::Vec3b>(i,j) = colors[index-1];
                else
                    dst.at<cv::Vec3b>(i,j) = cv::Vec3b(0,0,0);
            }
        }
        // Visualize the final image
        imshow("Final Result", dst);
	//****No FUNCIONA HASTA AQUI
        
        cv::waitKey(2000);
    }
  return 0;
}

/*for (int i = 0; i < contours.size(); ++i)
        {
          std::cout <<  "*************************************" << "\n" ;
          std::vector<cv::Point> v = contours[i];
          for (int j = 0; j < v.size(); ++j)
          {
            std::cout <<  "punto x es: "<< v[j].x <<  ", punto y es: "<< v[j].y << "\n" ;
          }
        }*/
