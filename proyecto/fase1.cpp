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
#include <string>

struct Region{
	int pixel_valueR;
	int pixel_valueG;
	int pixel_valueB;
	int votes;
};

/**
 * Esta funcion se encarga de detectar los puntos de interesa en la imagen original y en la mascara
 * @param image      imagen de entrada
 * @param imageMask  mascara
 * @param keypoints1 vector de puntos de la imagen de entrada
 * @param keypoints2 vector de puntos de la mascara
 */
void detectedPoints(cv::Mat image, cv::Mat imageMask, std::vector<cv::KeyPoint>& keypoints1, std::vector<cv::KeyPoint>& keypoints2){
    cv::SurfFeatureDetector detector(5000);

    //detecta los puntos de La dos imagenes
    detector.detect(image, keypoints1);
    detector.detect(imageMask, keypoints2);

    //objetos para visualizar los puntos en la imagen
    cv::Mat img_keypoints_1;
    cv::drawKeypoints( image, keypoints1, img_keypoints_1, cv::Scalar::all(-1), 0); //DrawMatchesFlags::DEFAULT

    //cv::imshow("Puntos de la imagen", img_keypoints_1 );
}

/**
 * Esta funcion se encarga de detectar los puntos de interesa en la imagen de entrada
 * @param image      imagen de entrada
 * @param keypoints1 vector de puntos de la imagen de entrada
 */
void detectedPoint(cv::Mat image, std::vector<cv::KeyPoint>& keypoints1){
    cv::SurfFeatureDetector detector(5000);

    //detecta los puntos de La dos imagenes
    detector.detect(image, keypoints1);

    //objetos para visualizar los puntos en la imagen
    cv::Mat img_keypoints_1;
    cv::drawKeypoints( image, keypoints1, img_keypoints_1, cv::Scalar::all(-1), 0); //DrawMatchesFlags::DEFAULT

    //cv::imshow("Puntos de la imagen", img_keypoints_1 );
}

/**
 * este metodo se encarga de la creacion de los descriptores
 * @param image       imagen
 * @param keypoints   vector de puntoa de interes 
 * @param descriptors el descriptore
 */
void createDescriptor(cv::Mat image, std::vector<cv::KeyPoint> keypoints ,cv::Mat& descriptors){
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
void createFLAN(cv::Mat descriptors1, cv::Mat descriptors2, std::vector< cv::DMatch >& matches, 
                double& max_dist, double& min_dist, std::vector< cv::DMatch >& good_matches,
                std::vector< cv::DMatch >& bad_matches ){

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
    { 
      if( matches[i].distance <= cv::max(2*min_dist, 0.002) ){ 
        good_matches.push_back( matches[i]); 
      }
      else{
        bad_matches.push_back( matches[i]); 
      }
    }
}


/**
 * encuentra el contorno y lo pinta con respecto a la imagen original
 * @param image     imagen original en escala de grises
 * @param contours  vector de contorno
 * @param hierarchy 
 */
void findContoursImage(cv::Mat image, std::vector<std::vector<cv::Point> >& contours, std::vector<cv::Vec4i>& hierarchy){
    cv::Canny(image, image, 100, 200, 3);
    cv::RNG rng(12345);
    cv::findContours( image, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0) );
}


/**
 * Este metodo es el encargado de leer las direcciones donde estas las mariposas
 * @param path        path donde estan las direcciones
 * @param directiones vector con las direciones
 */
void readPathTrain( std::string path, std::vector<std::string>& directiones) {
    std::ifstream file;
    std::string line;

    file.open(path.c_str(), std::ifstream::in);
    
    //Leer el archivo hasta que termine.
    while (!file.eof()) {
        getline(file, line); //Lee la linea
        directiones.push_back(line); //agrega la linea al vector
    }
    
}

/**
 * Este metodo es el encargado de leer las imagenes
 * @param directiones vector con las direcciones
 * @param images      vector con las imagenes originales
 * @param imagesMask  vector con las imagenes de mascara
 */
void readImage(std::vector<std::string> directiones, std::vector<cv::Mat>& images, std::vector<cv::Mat>& imagesMask){
    
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
	floodFillPostprocess( segmented_image, cv::Scalar::all(12) );
}

int vote(
	int		     pixel_valueR, 
	int		     pixel_valueG, 
	int		     pixel_valueB, 
	std::vector<Region>& regions)
{
	for(int i=0; i<regions.size(); i++){
		if ((regions.at(i).pixel_valueR==pixel_valueR) && (regions.at(i).pixel_valueG==pixel_valueG) && (regions.at(i).pixel_valueB==pixel_valueB) ){
			regions.at(i).votes+=1;
			return 0;
		}
	}
	return 1;
}

int getVotes(
	int pixel_valueR,
	int pixel_valueG,
	int pixel_valueB,
	std::vector<Region>& regions){
	for(int i=0; i<regions.size(); i++){
		if ((regions.at(i).pixel_valueR==pixel_valueR) && (regions.at(i).pixel_valueG==pixel_valueG) && (regions.at(i).pixel_valueB==pixel_valueB) ){
			return regions.at(i).votes;
		}
	}
	return 0;
}

int getMinMaxVotes(
	std::vector<Region>& regions,
	int& 		     max,
	int&		     min)
{
	for(int i=0; i<regions.size(); i++){
		if (regions.at(i).votes<min){
			min = regions.at(i).votes;
		}
		if (regions.at(i).votes>max){
			max = regions.at(i).votes;
		}
	}
}


void drawResults(
	cv::Mat&		   segmented_image,
	cv::Mat&		   result_image,
	std::vector<cv::KeyPoint>& image_detected_keypoints)
{
	int i, j, k, l;
	std::vector<Region> regions;
	for(i=0; i<image_detected_keypoints.size(); i++){
		cv::KeyPoint keypoint = image_detected_keypoints.at(i);
		int pixel_valueB = segmented_image.at<cv::Vec3b>(cv::Point(keypoint.pt.x, keypoint.pt.y))[0];
		int pixel_valueG = segmented_image.at<cv::Vec3b>(cv::Point(keypoint.pt.x, keypoint.pt.y))[1];
		int pixel_valueR = segmented_image.at<cv::Vec3b>(cv::Point(keypoint.pt.x, keypoint.pt.y))[2];
		if (vote(pixel_valueR, pixel_valueG, pixel_valueB,regions)){
			Region region;
			region.pixel_valueR = pixel_valueR;
			region.pixel_valueG = pixel_valueG;
			region.pixel_valueB = pixel_valueB;
			region.votes = 1;
			regions.push_back(region);
		}
				
	}
	
	for (j=0; j<regions.size(); j++){
		std::cout << "Pixel ValueR is: " << regions.at(j).pixel_valueR;
		std::cout << " Pixel ValueG is: " << regions.at(j).pixel_valueG;
		std::cout << " Pixel ValueB is: " << regions.at(j).pixel_valueB;
		std::cout << " Votes: " << regions.at(j).votes << std::endl;
	}
	
	int maxVotes = 0;
	int minVotes = 0;
	getMinMaxVotes(regions, maxVotes, minVotes);
	

	unsigned char *input = (unsigned char*)(result_image.data);
	unsigned char *inputSegmented = (unsigned char*)(segmented_image.data);
	for(int j = 0;j <result_image.rows;j++){
    		for(int i = 0;i < result_image.cols;i++){
			int pixel_valueB = segmented_image.at<cv::Vec3b>(cv::Point(i, j))[0];
			int pixel_valueG = segmented_image.at<cv::Vec3b>(cv::Point(i, j))[1];
			int pixel_valueR = segmented_image.at<cv::Vec3b>(cv::Point(i, j))[2];
			int votes = getVotes(pixel_valueR, pixel_valueG,pixel_valueB,regions);
			if (votes ==0){	
				result_image.at<cv::Vec3b>(cv::Point(i, j))[0] =(unsigned char)0;
				result_image.at<cv::Vec3b>(cv::Point(i, j))[1] =(unsigned char)0;
				result_image.at<cv::Vec3b>(cv::Point(i, j))[2] =(unsigned char)0;
			}
			else{
				int valor =  (int)((((float)votes-(float)minVotes)/((float)maxVotes-(float)minVotes))*150.0);
				result_image.at<cv::Vec3b>(cv::Point(i, j))[0] =(unsigned char)(valor+100);
				result_image.at<cv::Vec3b>(cv::Point(i, j))[1]= (unsigned char)(valor+100);
				result_image.at<cv::Vec3b>(cv::Point(i, j))[2]= (unsigned char)(valor+100);
			}
    		}
	}

	for(i=0; i<image_detected_keypoints.size(); i++){
		cv::KeyPoint keypoint = image_detected_keypoints.at(i);
		int pixel_valueB = segmented_image.at<cv::Vec3b>(cv::Point(keypoint.pt.x, keypoint.pt.y))[0];
		int pixel_valueG = segmented_image.at<cv::Vec3b>(cv::Point(keypoint.pt.x, keypoint.pt.y))[1];
		int pixel_valueR = segmented_image.at<cv::Vec3b>(cv::Point(keypoint.pt.x, keypoint.pt.y))[2];
		cv::Rect rect(keypoint.pt.x, keypoint.pt.y, 5, 5);
		cv::rectangle(result_image, rect, cv::Scalar(255,0,0),1,8,0);
				
	}
}

/**
 * Metodo encargado de realizar el entrenamiento
 * @param vectorsMaxDistance vector con todas las distancias maximas 
 * @param vectorsMinDistance vector con todas las distancias minimas
 * @param vectorsGoodMatches vector con todos los puntos que corresponden a una mariposa
 * @param vectorsBadMatches  vector con todos los puntos que no corresponden a una mariposa 
 */
/**
 * Metodo encargado de realizar el entrenamiento
 * @param vectorsMaxDistance vector con todas las distancias maximas 
 * @param vectorsMinDistance vector con todas las distancias minimas
 * @param vectorsGoodMatches vector con todos los puntos que corresponden a una mariposa
 * @param vectorsBadMatches  vector con todos los puntos que no corresponden a una mariposa 
 */
void train(
  std::vector< double > & vectorsMaxDistance,
  std::vector< double > & vectorsMinDistance,
  std::vector< cv::Mat > & vectorsDescriptors,
  std::vector< std::vector< cv::DMatch > > & vectorsGoodMatches,
  std::vector< std::vector< cv::DMatch > > & vectorsBadMatches
  ){

    std::string path = "../../../Database_of_Monarch_Butterflies/pathTrain.txt";
    std::vector<std::string> directiones;

    //lee el path donde estan las direciones.
    readPathTrain(path,directiones);

    //for (int i = 0; i < directiones.size(); ++i)
    //{
      //std::cout << directiones[i] << "\n";
    //}
    //
    //

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
        //************************************************
        //Detectar los puntos
        //Vector para los puntos de la imagen original y la mascara
        std::vector<cv::KeyPoint> keypoints1, keypoints2;
        //llamar a la funcion de detectar los puntos
        detectedPoints(image,imageMask, keypoints1, keypoints2);
        
        //************************************************
        //************************************************
        //************************************************
        //Descriptores
        //llamar a la funcion de descriptores.
        cv::Mat descriptors1, descriptors2;
        createDescriptor(image, keypoints1, descriptors1);
        createDescriptor(imageMask, keypoints2, descriptors2);
        
        vectorsDescriptors.push_back(descriptors2);
        /*for (int i = 0; i < keypoints1.size(); ++i)
        {
            float* descPtr = descriptors1.ptr<float>(i);
            for (int j = 0; j < descriptors1.cols; j++)
              std::cout  << *descPtr++ << " ";

            std::cout << "***********************\n";
        }*/
        
        //************************************************
        //************************************************
        //************************************************
        //FLAN y descriptor
        //vector con las coincidencias 
        std::vector< cv::DMatch > matches;
        double max_dist = 0; double min_dist = 100;
        
        //vector con los maches que realmente conincidieron
        std::vector< cv::DMatch > good_matches;
        std::vector< cv::DMatch > bad_matches;
        createFLAN(descriptors1, descriptors2, matches,max_dist, min_dist, good_matches, bad_matches);

        //printf("-- Maxima distancia : %f \n", max_dist );
        //printf("-- Minima dist : %f \n", min_dist );

        vectorsMaxDistance.push_back(max_dist);
        vectorsMinDistance.push_back(min_dist);

        vectorsGoodMatches.push_back(good_matches);
        vectorsBadMatches.push_back(bad_matches);

        //dibuja las coincidencias
        cv::Mat img_matches;
        cv::drawMatches( image, keypoints1, imageMask, keypoints2,
                     good_matches, img_matches, cv::Scalar::all(-1), cv::Scalar::all(-1),
                     std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

        //mustra las conincidencias
        //cv::imshow( "Coincidencias que cumplen", img_matches );

        //for( int i = 0; i < (int)good_matches.size(); i++ )
        //{ printf( "-- G [%d] Puntos de la imagen: %d  -- Puntos de la mascara: %d -- Distancia: %f \n", 
        //  i, good_matches[i].queryIdx, good_matches[i].trainIdx, good_matches[i].distance); }
        
        //************************************************
        //************************************************
        //************************************************
        //Para encontrar los contornos
        /*cv::Mat gray=image;
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
        }*/     
     
        //cv::imshow( "Resultado del contorno", drawing );
        
        //std::cout <<  descriptors1.rows << " " << descriptors1.cols<<"\n" ;
        
        cv::waitKey(2000);
    }
}
        

/**
 * Este metodo obtiene el promedio de la distacia maxima y minima
 * @param vectorsMaxDistance vector con todas las distancias maximas 
 * @param vectorsMinDistance vector con todas las distancias minimas
 * @param averageMaxDistance promedio de la distacia maxima
 * @param averageMinDistance promedio de la distancia minima
 */
void averageDistance(
    std::vector< double > vectorsMaxDistance,
    std::vector< double > vectorsMinDistance,
    double& averageMaxDistance, 
    double& averageMinDistance){

    double averageMaxTemporary;
    double averageMinTemporary;

    for (int i = 0; i < vectorsMaxDistance.size(); ++i)
    {
      averageMaxTemporary += vectorsMaxDistance[i];
      averageMinTemporary += vectorsMinDistance[i];
    }

    averageMaxTemporary = averageMaxTemporary/vectorsMaxDistance.size();
    averageMinTemporary = averageMinTemporary/vectorsMinDistance.size();

    averageMaxDistance = averageMaxTemporary;
    averageMinDistance = averageMinTemporary;

}



void testPoints(
    double averageMaxDistance,
    double averageMinDistance,
    std::vector< cv::Mat > vectorsDescriptors
    //std::vector< cv::KeyPoint > & vectorButterfly,
    //std::vector< cv::KeyPoint > & vectorNoButterfly
    ){

    std::string path = "../../../Database_of_Monarch_Butterflies/pathTest.txt";
    std::vector<std::string> directiones;

    //lee el path donde estan las direciones.
    readPathTrain(path,directiones);

    std::vector<cv::Mat> images; 
    std::vector<cv::Mat> imagesMask;

    //lee las imagenes de originales y las mascaras
    readImage(directiones,images,imagesMask);

    for (int i = 0; i < images.size(); ++i)
    {
        cv::Mat image = images[i];
	imshow("Image In", image);
        cv::Mat imageMask = imagesMask[i];

        std::vector<int> vectorButterfly;
        std::vector<int> vectorNoButterfly;
        std::vector< cv::KeyPoint > Butterfly;
	std::vector< cv::KeyPoint > noButterfly;
        //************************************************
        //************************************************
        //************************************************
        //Detectar los puntos
        //Vector para los puntos de la imagen original y la mascara
        std::vector<cv::KeyPoint> keypoints1;
        //llamar a la funcion de detectar los puntos
        detectedPoint(image, keypoints1);
        
        //************************************************
        //************************************************
        //************************************************
        //Descriptores
        //llamar a la funcion de descriptores.
        cv::Mat descriptors1;
        createDescriptor(image, keypoints1, descriptors1);
        

        //************************************************
        //************************************************
        //************************************************
        //FLAN y descriptor
        //vector con las coincidencias 
        std::vector< cv::DMatch > matches;
        std::vector< cv::DMatch > good_matches;
        std::vector< cv::DMatch > bad_matches;

        for (int j = 0; j < vectorsDescriptors.size(); ++j)
        {
          cv::Mat trainDescriptors = vectorsDescriptors[j];

          cv::FlannBasedMatcher matcher;

          //obtine los match
          matcher.match( descriptors1, trainDescriptors, matches );

          //obtengo las coincidencias que realmente realizan los maches cumpliendo entre el rango establecido
          for( int i = 0; i < descriptors1.rows; i++ )
          { 
            if( matches[i].distance <= cv::max(2*averageMinDistance, 0.002) ){ 
              good_matches.push_back(matches[i]);
              bool flag= false;
              for (int k = 0; k < vectorButterfly.size(); ++k)
              {
                if (i!=vectorButterfly.at(k))
                {
                  flag = false;
                }
                else{
                  flag = true;
                  break;
                }
              }
              if (!flag)
              {
                vectorButterfly.push_back(i);
              }
            }
            else{
              bad_matches.push_back( matches[i]);
              bool flag= false;
              for (int k = 0; k < vectorNoButterfly.size(); ++k)
              {
                if (i!=vectorNoButterfly.at(k))
                {
                  flag = false;
                }
                else{
                  flag = true;
                  break;
                }
              }
              if (!flag)
              {
                vectorNoButterfly.push_back(i);
              }

            }
          }
        }// fin de los maches del descriptor


        /*for (int i = 0; i < vectorButterfly.size(); ++i)
        {
          std::cout << "La posicion es: " <<vectorButterfly[i] << " ";
        }
        std::cout << "*********************" << "\n";
        std::cout << "*********************" << "\n";
        std::cout << "*********************" << "\n";*/
        
       for (int i = 0; i < vectorNoButterfly.size(); ++i)
       {
          int position = vectorNoButterfly.at(i); 
          Butterfly.push_back(keypoints1[position]);
       }
       for (int i = 0; i < vectorButterfly.size(); ++i)
       {
          int position = vectorButterfly.at(i); 
          noButterfly.push_back(keypoints1[position]);
       }

       std::cout << "el largo es  " <<Butterfly.size() << "\n";
       //
       //***************************************************************
       //***************************************************************
       //***************************************************************
       //***************************************************************
       //ACA SIGUE SU CODIGO POR LA IMAGEN CARGADA YA
       //EL VECTOR BUTTERFLY ES EL QUE TIENE LOS SUPUESTOS PUNTOS QUE SON MARIPOSA
       //LISTOS PARA VOTAR PARA CADA IMAGEN
       // ESOS PUNTOS SON POR CADA IMAGEN QUE SUBE DE TEST
       // NO SE DONDE ESPECIFICAMENTE DONDE ESTAN UBICADOS PERO CREO QUE NO SE ESTA TAN PERDIDOS.
	cv::Mat segmented_image;      
	meanShiftSegmentation(image, segmented_image); 
	cv::Mat result(segmented_image);   
	imshow("segmented", segmented_image);
	drawResults(segmented_image, result, Butterfly);
	imshow("Result", result);
       cv::waitKey(2000);
    }
}


int main(int argc, char** argv){

    std::vector< double > vectorsMaxDistance;
    std::vector< double > vectorsMinDistance;
    std::vector< cv::Mat > vectorsDescriptors;
    std::vector< std::vector< cv::DMatch > > vectorsGoodMatches;
    std::vector< std::vector< cv::DMatch > > vectorsBadMatches;

    train(vectorsMaxDistance, vectorsMinDistance, vectorsDescriptors, vectorsGoodMatches, vectorsBadMatches);

    /*for (int j = 0; j < vectorsGoodMatches.size(); ++j)
    {
      std::cout <<  "---------------------------------------------------------------------------------------------" << "\n" ;
      std::vector< cv::DMatch > good_matches = vectorsGoodMatches[j];
      for( int i = 0; i < (int)good_matches.size(); i++ )
        { printf( "-- G [%d] Puntos de la imagen: %d  -- Puntos de la mascara: %d -- Distancia: %f \n", 
          i, good_matches[i].queryIdx, good_matches[i].trainIdx, good_matches[i].distance); }
    }*/

    // obtine el promedio de la distncias del entrenamiento para obtener el umbral.
    double averageMaxDistance;
    double averageMinDistance; 
    averageDistance(vectorsMaxDistance, vectorsMinDistance, averageMaxDistance, averageMinDistance);
    testPoints(averageMaxDistance,averageMinDistance,vectorsDescriptors);
    
  return 0;
}


