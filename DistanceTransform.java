/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package bik;

import bik.*;
import static org.bytedeco.javacpp.opencv_core.*;
import static org.bytedeco.javacpp.opencv_highgui.*;
import static org.bytedeco.javacpp.opencv_imgproc.*;

import java.awt.Point;
import java.io.*;
import java.util.*;
import java.text.DecimalFormat;
import java.util.Arrays;
import java.util.concurrent.*;

import image_processing.CCExtraction;
import image_processing.CCImage;
import image_processing.Utilities;
import image_processing.Zhang2008;
import java.nio.FloatBuffer;

/**
 *
 * @author adapt
 */
public class DistanceTransform {

    public static int K = -1;
    IplImage imageOriginal;
    IplImage imageNormalized;
    IplImage dtImage;
    public List<Keypoint> keypointList = new ArrayList<>();
    public List<Integer> keypointFDistanceList = new ArrayList<>();
    public double foreground_point_count = 0;
    public double point_reduction_rate = 0;
    List<Point> originalForegroundPointList = new ArrayList<>();
    List<Point> shiftedFregroundPointList = new ArrayList<>();
//    List<Point> foregroundPointWithoutAdditionalBordersList = new ArrayList<>();    // used for keypoint feature extraction
    public Point centroid_original = new Point();
    public Point centroid_shifted = new Point();
    public double contour_point_count = 0;
    public static final boolean SHIFT_BACK_KEYPOINT_LOCATIONS = false;

    public DistanceTransform(IplImage image) {

//        imageOriginal = Utilities.getContourImage(image);

        this.imageOriginal = image.clone();

        double x = 0;
        double y = 0;
        for (int i = 0; i < image.height(); i++) {
            for (int j = 0; j < image.width(); j++) {
                Point p = new Point(i, j);
                if (Utilities.isBlack(p, image)) {
                    foreground_point_count++;
                    x += p.x;
                    y += p.y;
                    if (Utilities.isContourPoint(p, image)) {
                        originalForegroundPointList.add(p);
                    }
                }
            }
        }
        x /= foreground_point_count;
        y /= foreground_point_count;
        centroid_original.setLocation(x, y);

//        cvSet2D(image, centroid_original.x, centroid_original.y, CvScalar.BLACK);
//        System.out.println("centroid_original = " + centroid_original);
//        cvCircle(image, cvPoint(centroid_original.y, centroid_original.x), 10, CvScalar.BLACK, 0, 0, 0);
//        cvShowImage("original", image);

//        this.imageNormalized = Utilities.applyBoundingBoxNormalization(Utilities.getContourImage(image));
        this.imageNormalized = Utilities.applyBoundingBoxNormalization(imageOriginal);
//        System.out.println("image normalized.");
        x = 0;
        y = 0;
        for (int i = 0; i < imageNormalized.height(); i++) {
            for (int j = 0; j < imageNormalized.width(); j++) {
                Point p = new Point(i, j);
                if (Utilities.isBlack(p, imageNormalized)) {
                    x += p.x;
                    y += p.y;
                }
            }
        }
        x /= foreground_point_count;
        y /= foreground_point_count;
        centroid_shifted.setLocation(x, y);

//        for (Point p : shiftedFregroundPointList) {
//            Point p_shifted = Utilities.getShiftedPoint(p, centroid_shifted, centroid_original);
//            originalForegroundPointList.add(p_shifted);
//        }



//        foregroundPointWithoutAdditionalBordersList.addAll(shiftedFregroundPointList);
//        foregroundPointWithoutAdditionalBordersList.addAll(originalForegroundPointList);



//        cvSet2D(imageNormalized, centroid_shifted.x, centroid_shifted.y, CvScalar.BLACK);
//        cvCircle(imageNormalized, cvPoint(centroid_shifted.y, centroid_shifted.x), 10, CvScalar.BLACK, 0, 0, 0);
//        System.out.println("centroid_shifted = " + centroid_shifted);
//        cvShowImage("shifted", imageNormalized);
//        cvWaitKey(0);


        // make contour image
//        Zhang2008 thinning = new Zhang2008();
//        thinning.setImage(imageNormalized.clone());
//        IplImage image_contour = thinning.getThinnedImage();
        IplImage image_contour = Utilities.getContourImage(imageNormalized);
        for (int i = 1; i < imageNormalized.height() - 1; i++) {
            for (int j = 1; j < imageNormalized.width() - 1; j++) {
                Point p = new Point(i, j);
                if (Utilities.isBlack(p, image_contour)) {
//                    if (Utilities.isContourPoint(p, image) == false) {
                    cvSet2D(imageNormalized, i, j, CvScalar.BLACK);
                    contour_point_count++;
                    shiftedFregroundPointList.add(p);
//                    foregroundPointWithoutAdditionalBordersList.add(p);
//                    }
                } else {
                    cvSet2D(imageNormalized, i, j, CvScalar.WHITE);
                }
            }
        }

        // set borders in black
        for (int i = 0; i < imageNormalized.height(); i++) {
            cvSet2D(imageNormalized, i, 0, CvScalar.BLACK);
            cvSet2D(imageNormalized, i, imageNormalized.width() - 1, CvScalar.BLACK);

//            originalForegroundPointList.add(new Point(i, 0));
//            originalForegroundPointList.add(new Point(i, imageNormalized.width() - 1));
        }
        for (int j = 0; j < imageNormalized.width(); j++) {
            cvSet2D(imageNormalized, 0, j, CvScalar.BLACK);
            cvSet2D(imageNormalized, imageNormalized.height() - 1, j, CvScalar.BLACK);
//            originalForegroundPointList.add(new Point(0, j));
//            originalForegroundPointList.add(new Point(imageNormalized.height() - 1, j));
        }
        dtImage = imageNormalized.clone();

//        CCExtraction extract = new CCExtraction(dtImage);
//        extract.extractCC();
//        System.out.println("CC extraction done.");
//        for (List<Point> list : extract.CC_list) {
//            originalForegroundPointList.addAll(list);
//        }
    }

    private void calculateDTImageFast() {
        IplImage binary = Utilities.binarize(imageNormalized);
//        cvShowImage("binary", binary);
        cvDistTransform(binary, binary, CV_DIST_L1, 0, (FloatBuffer) null, null, 0);
        dtImage = binary;

        // normalization using the max pixel value
        int distance_max = 1;
        for (int i = 0; i < dtImage.height(); i++) {
            for (int j = 0; j < dtImage.width(); j++) {
                if (!Utilities.isBlack(new Point(i, j), dtImage)) {
                    double value = cvGet2D(dtImage, i, j).magnitude();
                    if (distance_max < value) {
                        distance_max = (int) value;
                    }
                }
            }
        }
        for (int i = 0; i < dtImage.height(); i++) {
            for (int j = 0; j < dtImage.width(); j++) {
                if (!Utilities.isBlack(new Point(i, j), dtImage)) {
                    double value = cvGet2D(dtImage, i, j).magnitude();
                    value = value * 255 / distance_max;
                    CvScalar value_normalized = cvScalar(value, value, value, value);
                    cvSet2D(dtImage, i, j, value_normalized);
                }
            }
        }
    }

    // This method is stable and far less time-consuming than using growing circles
//    public int getPointFDistance(Point p) {
//        double distance_min = Double.MAX_VALUE;
//        for (Point q : originalForegroundPointList) {
//            double distance = Utilities.euclidean(q, p);
//            if (distance < distance_min) {
//                distance_min = distance;
//            }
//        }
//
//        return (int) distance_min;
//    }

    public Point findNearestFGPoint(Point p) {
        double distance_min = Double.MAX_VALUE;
        Point nearest_point = new Point();

        List<Point> list;

        if (SHIFT_BACK_KEYPOINT_LOCATIONS) {
            list = originalForegroundPointList;
        } else {
            list = shiftedFregroundPointList;
        }

        for (Point q : list) {
            double distance = Utilities.euclidean(q, p);
            if (distance < distance_min) {
                distance_min = distance;
                nearest_point = q;
            }
        }

        return nearest_point;
    }

    // Detects the local maxima in dtImage
    private List<Point> getLocalMaxima() {
        
        List<Point> local_maxima_list = new ArrayList<>();

        int MARGE = K/2;
        
        for (int i = MARGE; i < dtImage.height() - MARGE; i++) {
            for (int j = MARGE; j < dtImage.width() - MARGE; j++) {
                if (!Utilities.isBlack(new Point(i, j), imageNormalized)) {
                    boolean isMaxima = true;
                    double value_i_j = cvGet2D(dtImage, i, j).magnitude();
                    for (int x = i - MARGE; x <= i + MARGE; x++) {
                        for (int y = j - MARGE; y <= j + MARGE; y++) {
                            double value_x_y = cvGet2D(dtImage, x, y).magnitude();
                            if (value_i_j < value_x_y) {
                                isMaxima = false;
                            }
                        }
                    }
                    if (isMaxima) {
                        local_maxima_list.add(new Point(i, j));
                    }
                }
            }
        }

//        System.out.println("Local Maxima detected.");

        return local_maxima_list;
    }

    // Detects keypoints which are the centroid of the components in the contour map image
    public void detectKeypoints() {

        // first, calculate the DT image
        calculateDTImageFast();
//        System.out.println("DT calculated.");
//        cvShowImage("DT", dtImage);

        List<Point> local_maxima_list = getLocalMaxima();

        // draw the contour map
        IplImage contour_map = IplImage.create(imageNormalized.cvSize(), 8, 3);
//        IplImage contour_map = imageNormalized.clone();
        for (int i = 0; i < contour_map.height(); i++) {
            for (int j = 0; j < contour_map.width(); j++) {
                cvSet2D(contour_map, i, j, CvScalar.WHITE);
            }
        }
        for (Point p : local_maxima_list) {
            cvSet2D(contour_map, p.x, p.y, CvScalar.BLACK);
        }
//        System.out.println("Contour map calculated.");
//        cvShowImage("contour map", contour_map);
//        cvWaitKey(0);

//        contour_map = Utilities.binaryToRGB(contour_map);
//        cvSaveImage("binary.png", contour_map);
//        contour_map = cvLoadImage("binary.png");
        CCExtraction extract = new CCExtraction(contour_map);
        extract.extractCC();
        CCImage cci = extract.getCCImage();
//        System.out.println("cci extracted.");
//        cvShowImage("cci", cci.getImage());
//        cvWaitKey(0);
//        cvSaveImage("map.png", cci.getImage());

        // find the keypoints
        for (List<Point> component : cci.CC_list) {
            double x_mean = 0;
            double y_mean = 0;
            for (Point p : component) {
                x_mean += p.x;
                y_mean += p.y;
            }
            x_mean /= component.size();
            y_mean /= component.size();
            Point centroid = new Point();
            centroid.setLocation(x_mean, y_mean);

            // configure the keypoint
            Keypoint keypoint = new Keypoint(centroid);
            keypoint.nearestFgPoint = findNearestFGPoint(keypoint.location);
            keypoint.distance = (int) Utilities.euclidean(keypoint.location, keypoint.nearestFgPoint);
            keypointList.add(keypoint);
        }
//        System.out.println("Keypoints extracted.");

        // calculate the point reduction rate
        if (foreground_point_count > 0) {
//            point_reduction_rate = (keypointList.size()) * 100 / foreground_point_count;
            point_reduction_rate = (keypointList.size()) * 100 / contour_point_count;
            DecimalFormat df = new DecimalFormat("#.##");
            point_reduction_rate = Double.valueOf(df.format(point_reduction_rate));
        } else {
            point_reduction_rate = -1;
        }

        // extract keypoints features
        for (Keypoint keypoint : keypointList) {
//            keypoint.calculateFeatureVector(foregroundPointWithoutAdditionalBordersList);
            keypoint.calculateFeatureVector(shiftedFregroundPointList);
        }
//        System.out.println("features extracted.");

        // set back the location of the keypoints
        if (SHIFT_BACK_KEYPOINT_LOCATIONS) {
            for (int i = 0; i < keypointList.size(); i++) {

                Keypoint keypoint = keypointList.get(i);

//                int x_shifted = keypoint.location.x + (centroid_original.x - centroid_shifted.x);
//                int y_shifted = keypoint.location.y + (centroid_original.y - centroid_shifted.y);
//                boolean condition_1 = (x_shifted >= 0 && x_shifted < imageOriginal.height() && y_shifted >= 0 && y_shifted < imageOriginal.width());
//
//                x_shifted = keypoint.nearestFgPoint.x + (centroid_original.x - centroid_shifted.x);
//                y_shifted = keypoint.nearestFgPoint.y + (centroid_original.y - centroid_shifted.y);
//                boolean condition_2 = (x_shifted >= 0 && x_shifted < imageOriginal.height() && y_shifted >= 0 && y_shifted < imageOriginal.width());

                keypoint.location = Utilities.getShiftedPoint(keypoint.location, centroid_original, centroid_shifted);
//                keypoint.nearestFgPoint = Utilities.getShiftedPoint(keypoint.nearestFgPoint, centroid_shifted, centroid_original);
                keypoint.nearestFgPoint = findNearestFGPoint(keypoint.location);
                keypoint.distance = (int) Utilities.euclidean(keypoint.location, keypoint.nearestFgPoint);

                boolean condition_1 = (keypoint.location.x >= 0 && keypoint.location.x < imageOriginal.height() && keypoint.location.y >= 0 && keypoint.location.y < imageOriginal.width());
//                boolean condition_2 = (keypoint.nearestFgPoint.x >= 0 && keypoint.nearestFgPoint.x < imageOriginal.height() && keypoint.nearestFgPoint.y >= 0 && keypoint.nearestFgPoint.y < imageOriginal.width());
                
                keypointList.set(i, keypoint);
                
                // set the keypoint location
//                keypoint.location.setLocation(x_shifted, y_shifted);
                // set the keypoint nearest FG location
//                keypoint.nearestFgPoint.setLocation(x_shifted, y_shifted);
                if (!condition_1) {
//                if (!condition_1 || !condition_2) {
                    keypointList.remove(i);
                    i--;
//                    System.out.println(keypoint.location + " has been removed.");
                }
            }
        }

    }

    public IplImage getHighlightedImage() {

        IplImage image;
        if (SHIFT_BACK_KEYPOINT_LOCATIONS) {
//            image = imageOriginal.clone();
            image = IplImage.create(imageOriginal.cvSize(), 8, 3);
            for (int i = 0; i < image.height(); i++) {
                for (int j = 0; j < image.width(); j++) {
                    if (Utilities.isBlack(new Point(i, j), imageOriginal)) {
                        cvSet2D(image, i, j, CvScalar.BLACK);
                    } else {
                        cvSet2D(image, i, j, CvScalar.WHITE);
                    }
                }
            }
        } else {
//            image = imageNormalized.clone();
//            IplImage image_normalized = Utilities.applyBoundingBoxNormalization(imageOriginal.clone());
            IplImage image_normalized = imageNormalized;
            image = IplImage.create(image_normalized.cvSize(), 8, 3);
            for (int i = 0; i < image.height(); i++) {
                for (int j = 0; j < image.width(); j++) {
                    if (Utilities.isBlack(new Point(i, j), image_normalized)) {
                        cvSet2D(image, i, j, CvScalar.BLACK);
                    } else {
                        cvSet2D(image, i, j, CvScalar.WHITE);
                    }
                }
            }
        }
        // highlight keypoints with circles and arrows
        for (Keypoint keypoint : keypointList) {
            // find the nearest foreground point
//            double distance_min = Double.MAX_VALUE;
//            Point nearest_fg = new Point();
//            for (int i = 0; i < image.height(); i++) {
//                for (int j = 0; j < image.width(); j++) {
//                    Point p = new Point(i, j);
//                    if (Utilities.isBlack(p, image)) {
//                        double distance = Utilities.euclidean(p, keypoint.location);
//                        if (distance < distance_min) {
//                            distance_min = distance;
//                            nearest_fg = p;
//                        }
//                    }
//                }
//            }

            Point nearest_fg = keypoint.nearestFgPoint;

            // draw a circle and and arrow
            cvSet2D(image, keypoint.location.x, keypoint.location.y, CvScalar.RED);
            cvCircle(image, cvPoint(keypoint.location.y, keypoint.location.x), keypoint.distance, CvScalar.RED, 0, 0, 0);

            Utilities.drawArrow(image, cvPoint(keypoint.location.y, keypoint.location.x), cvPoint(nearest_fg.y, nearest_fg.x), CV_RGB(100, 0, 255), 5, 1, 8, 0);

        }

        return image;
    }

    public static void main(String args[]) {

//        highlight();

        IplImage image_big = cvLoadImage("Kbig.png");




//        image_big = Utilities.applyBoundingBoxNormalization(image_big);

        int size_factor = 2;
        CvSize size = cvSize(image_big.width() / size_factor, image_big.height() / size_factor);
        IplImage image = IplImage.create(size, image_big.depth(), image_big.nChannels());
        cvResize(image_big, image);

        // leave only the contour pixels
//        IplImage image_contour = image.clone();
//        for (int i = 1; i < image.height() - 1; i++) {
//            for (int j = 1; j < image.width() - 1; j++) {
//                Point p = new Point(i, j);
//                if (Utilities.isBlack(p, image)) {
//                    if (Utilities.isContourPoint(p, image) == false) {
//                        cvSet2D(image_contour, i, j, CvScalar.WHITE);
//                    }
//                }
//            }
//        }
//        cvShowImage("contour", image_contour);
//        cvWaitKey(0);
//        image = image_contour;

//        DistanceTransform dt = new DistanceTransform(image_contour);
        DistanceTransform.K = 15;
        DistanceTransform dt = new DistanceTransform(image);
//        dt.calculateDTImageFast();
        dt.detectKeypoints();
        if (!DistanceTransform.SHIFT_BACK_KEYPOINT_LOCATIONS) {
            image = dt.imageNormalized;
//            image = image;
        }
        cvShowImage("dt", dt.dtImage);
        cvWaitKey(0);
        cvSaveImage("dt.png", dt.dtImage);

        List<Keypoint> keypoint_list = dt.keypointList;
        for (Keypoint kp : keypoint_list) {
            cvSet2D(image, kp.location.x, kp.location.y, CvScalar.RED);
            cvCircle(image, cvPoint(kp.location.y, kp.location.x), 1, CvScalar.RED, -1, 0, 0);
        }
        cvShowImage("original", image);
        cvWaitKey(0);
        cvSaveImage("result.png", image);

        // calculate features
        for (Keypoint kp : keypoint_list) {
            kp.calculateFeatureVector(dt.originalForegroundPointList);
        }

        int n = keypoint_list.size();
        System.out.println(n + " keypoints detected.");

        image = dt.getHighlightedImage();
        cvShowImage("original", image);
        cvWaitKey(0);
        cvSaveImage("result.png", image);
        cvSaveImage("dt.png", dt.dtImage);
        cvSaveImage("normalized.png", dt.imageNormalized);

    }

    public static void highlight() {
        // load descriptors
//        List<String> descriptor_tab = new File(folder_str).list(new FilenameFilter() {
        String folder_str = "Dataset2";
        File dir = new File(folder_str);
        ArrayList<File> files = new ArrayList<>(Arrays.asList(dir.listFiles(new FilenameFilter() {
            @Override
            public boolean accept(File dir, String name) {
                return name.toLowerCase().endsWith(".png") || name.toLowerCase().endsWith(".tif") || name.toLowerCase().endsWith(".TIF");
            }
        })));
        Collections.sort(files);
        int image_index = 1;
        for (File file : files) {

            String image_str = file.getName();
            IplImage image = cvLoadImage(folder_str + "/" + image_str);

            DistanceTransform dt = new DistanceTransform(image);
            dt.calculateDTImageFast();

            dt.detectKeypoints();
            List<Keypoint> centroid_list = dt.keypointList;
            for (Keypoint kp : centroid_list) {
                cvSet2D(image, kp.location.x, kp.location.y, CvScalar.RED);
                cvCircle(image, cvPoint(kp.location.y, kp.location.x), 1, CvScalar.RED, -1, 0, 0);
            }
            int n = centroid_list.size();
            System.out.println("[" + image_index + "/ " + files.size() + "] " + image_str + " : " + n + " keypoints extracted.");

            cvSaveImage(folder_str + "/" + image_str, image);

            image_index++;
        }
    }
}
