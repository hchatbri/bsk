/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package bsk;

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

/**
 *
 * @author houssem
 */
public class bsk implements java.io.Serializable {

    public List<Keypoint> keypointList;
    transient public IplImage image_original;
    transient public List<IplImage> distortedImageList = new ArrayList<>();
    public static double STABILITY_THRESHOLD = 0;
    public double original_number_of_points = 0;
    public static final int NUMBER_OF_ITERATIONS = 5;
    public static double KEYPOINT_MAXIMUM_LOCATION_SHIFT = -1; //Math.sqrt(2);
    transient public IplImage image_normalized;
    public int sparcityIndex = 0;
    public static int N;
//    public Point centroid = new Point();
//    public double bounding_box_diameter;
    public int regression_index = -1;

    public bsk(IplImage image) {
        this.image_original = image;

        // find the centroid and bounding box diameter
//        double centroid_X = 0;
//        double centroid_Y = 0;
//        int number_of_fg_points = 0;
//        for (int i = 0; i < image.height(); i++) {
//            for (int j = 0; j < image.width(); j++) {
//                Point p = new Point(i, j);
//                if (Utilities.isBlack(p, image)) {
//                    centroid_X += p.x;
//                    centroid_Y += p.y;
//                    number_of_fg_points++;
//                }
//            }
//        }
//        centroid_X /= number_of_fg_points;
//        centroid_Y /= number_of_fg_points;
//        centroid.setLocation(centroid_X, centroid_Y);
    }

    public void performbsk() {
        DistanceTransform dt = new DistanceTransform(image_original);
        dt.detectKeypoints();
        image_normalized = dt.imageNormalized;
//        bounding_box_diameter = Math.sqrt(image_normalized.height() * image_normalized.height() + image_normalized.width() * image_normalized.width());
//        keypointList = new ArrayList<>();
        keypointList = dt.keypointList;
        original_number_of_points = dt.contour_point_count;

        // calculate keypoint scores
//        double distance_mean = 0;
//        for (Keypoint k1 : keypointList) {
//            for (Keypoint k2 : keypointList) {
//                distance_mean += Utilities.euclidean(k1.location, k2.location);
//            }
//        }
//        distance_mean /= keypointList.size() * keypointList.size();
//        distance_mean /= 2;
        for (Keypoint keypoint : keypointList) {
//            keypoint.calculateScoreDynamic(keypointList, distance_mean);
            keypoint.calculateScore(keypointList);
        }
        Collections.sort(keypointList, new Comparator<Keypoint>() {
            @Override
            public int compare(Keypoint k_1, Keypoint k_2) {
//                return (int)(k_1.score - k_2.score);
                return Double.compare(k_2.score, k_1.score);
            }
        });
    }

    public void performRegression() {

        double x_array[] = new double[keypointList.size()];
        double y_array[] = new double[keypointList.size()];

        double score_cumul = 0;
        for (int index = 0; index < keypointList.size(); index++) {

            x_array[index] = index + 1;
            y_array[index] = Math.log(1 + score_cumul);
//            y_array[index] = score_cumul;

            score_cumul += keypointList.get(index).score;
        }

        SegmentedRegression regress = new SegmentedRegression(x_array, y_array);
//        regress.performTwoSegmentsRegression();
        regress.performThreeSegmentsRegression();
        regression_index = regress.regression_x_index;
    }

    public void calculateSparcity() {

        List<Double> sparcityList = new ArrayList<>();

        for (int i = 0; i < keypointList.size(); i++) {
            double distance_mean = 0;
            for (int j = 0; j <= i; j++) {
                for (int k = 0; k <= i; k++) {
                    distance_mean += Utilities.euclidean(keypointList.get(j).location, keypointList.get(k).location);
                }
            }
            distance_mean /= (i + 1) * (i + 1);
            sparcityList.add(distance_mean);
        }

        for (int i = 0; i < sparcityList.size(); i++) {
            if (sparcityList.get(sparcityIndex) < sparcityList.get(i)) {
                sparcityIndex = i;
            }
        }

//        // draw curve
//        try {
//            File file = new File("data_gnuplot.txt");
//            BufferedWriter writer = new BufferedWriter(new FileWriter(file));
//            for (int index = 0; index < keypointList.size(); index++) {
//                String str = (index + 1) + " " + sparcityList.get(index);
//                writer.write(str);
//                writer.newLine();
//                writer.flush();
//            }
//            writer.close();
//            Runtime.getRuntime().exec("gnuplot plot.plt").waitFor();
//            IplImage plot = cvLoadImage("plot.png");
//            cvShowImage("window", plot);
//            cvWaitKey(0);
//        } catch (Exception e) {
//            e.printStackTrace();
//        }
    }

//    public void getKeypointsMaximizingSparcity() {
//
//        int index_max = 0;
//        for (int i = 0; i < sparcityList.size(); i++) {
//            if (sparcityList.get(index_max) < sparcityList.get(i)) {
//                index_max = i;
//            }
//        }
//        keypointMaximizingSparcityList = keypointList.subList(0, index_max);
//        int n = keypointMaximizingSparcityList.size();
//        System.out.println("Number of sparcity maximizing keypoints = " + n);
//    }
    public void showKeypointDescendingScore() {
        IplImage image = IplImage.create(image_normalized.cvSize(), 8, 3);
        for (int i = 0; i < image.height(); i++) {
            for (int j = 0; j < image.width(); j++) {
                if (Utilities.isBlack(new Point(i, j), image_normalized)) {
                    cvSet2D(image, i, j, CvScalar.BLACK);
                } else {
                    cvSet2D(image, i, j, CvScalar.WHITE);
                }
            }
        };


        for (int i = 0; i < keypointList.size(); i++) {

            Keypoint keypoint = keypointList.get(i);

            if (i < N) {
                cvSet2D(image, keypoint.location.x, keypoint.location.y, CvScalar.RED);
                cvCircle(image, cvPoint(keypoint.location.y, keypoint.location.x), 1, CvScalar.RED, -1, 1, 1);
            } else {
                cvSet2D(image, keypoint.location.x, keypoint.location.y, CvScalar.BLUE);
                cvCircle(image, cvPoint(keypoint.location.y, keypoint.location.x), 1, CvScalar.BLUE, -1, 1, 1);
            }
            cvShowImage("window", image);
            cvWaitKey(0);
        }
    }

    public IplImage showKeypoints() {

        IplImage image = IplImage.create(image_normalized.cvSize(), 8, 3);
        for (int i = 0; i < image.height(); i++) {
            for (int j = 0; j < image.width(); j++) {
                if (Utilities.isBlack(new Point(i, j), image_normalized)) {
                    cvSet2D(image, i, j, CvScalar.BLACK);
                } else {
                    cvSet2D(image, i, j, CvScalar.WHITE);
                }
            }
        };


        for (int i = 0; i < Math.min(N, keypointList.size()); i++) {
            Keypoint keypoint = keypointList.get(i);
            cvSet2D(image, keypoint.location.x, keypoint.location.y, CvScalar.RED);
            cvCircle(image, cvPoint(keypoint.location.y, keypoint.location.x), 2, CvScalar.RED, -1, 1, 1);
        }
        
        return image;
    }

    public static double match(bsk bsk_1, bsk bsk_2) {

//        final List<Keypoint> bsk_1_keypoint_list = new ArrayList<>();
////        for (int i = 0; i < bsk_1.keypointList.size(); i++) {
//        for (int i = 0; i < bsk_1.N; i++) {
//            if (bsk_1.keypointList.get(i).stability >= bsk.STABILITY_THRESHOLD) {
//                bsk_1_keypoint_list.add(bsk_1.keypointList.get(i));
//            }
//        }
//        final List<Keypoint> bsk_2_keypoint_list = new ArrayList<>();
////        for (int i = 0; i < bsk_2.keypointList.size(); i++) {
//        for (int i = 0; i < bsk_2.N; i++) {
//            if (bsk_2.keypointList.get(i).stability >= bsk.STABILITY_THRESHOLD) {
//                bsk_2_keypoint_list.add(bsk_2.keypointList.get(i));
//            }
//        }

        final double distance_tab[] = new double[N];
        for (int i = 0; i < distance_tab.length; i++) {
            distance_tab[i] = 0;
        }

        for (int i = 0; (i < N) && (i < bsk_1.keypointList.size()); i++) {

            double distance_min = Double.MAX_VALUE;

            for (int j = 0; (j < N) && (j < bsk_2.keypointList.size()); j++) {

                double distance = Keypoint.getHistogramDistance(bsk_1.keypointList.get(i), bsk_2.keypointList.get(j));

                if (distance < distance_min) {
                    distance_min = distance;
                }
            }

            distance_tab[i] = distance_min;
        }
        double distance_cumul = 0;
        for (double distance : distance_tab) {
            distance_cumul += distance;
        }
        distance_cumul /= Math.min(N, bsk_1.keypointList.size());

        return 1 / (distance_cumul + 1);
    }

    public static double matchSymmetric(bsk bsk_1, bsk bsk_2) {

        double value = (match(bsk_1, bsk_2) + match(bsk_2, bsk_1)) / 2;

        return value;
    }

    public static double matchMirrorInvariant(bsk bsk_1, bsk bsk_2) {

        final List<Keypoint> bsk_1_keypoint_list = new ArrayList<>();
        for (int i = 0; i < bsk_1.keypointList.size(); i++) {
            if (bsk_1.keypointList.get(i).stability >= bsk.STABILITY_THRESHOLD) {
                bsk_1_keypoint_list.add(bsk_1.keypointList.get(i));
            }
        }
//        int n = bsk_1_keypoint_list.size();
//        System.out.println("bsk_1 contains " + n + " keypoints.");

        final List<Keypoint> bsk_2_keypoint_list = new ArrayList<>();
        for (int i = 0; i < bsk_2.keypointList.size(); i++) {
            if (bsk_2.keypointList.get(i).stability >= bsk.STABILITY_THRESHOLD) {
                bsk_2_keypoint_list.add(bsk_2.keypointList.get(i));
            }
        }
//        n = bsk_2_keypoint_list.size();
//        System.out.println("bsk_2 contains " + n + " keypoints.");

        // compare with the original
        final double distance_tab[] = new double[bsk_1_keypoint_list.size()];
        for (int i = 0; i < distance_tab.length; i++) {
            distance_tab[i] = 0;
        }
        for (int i = 0; i < bsk_1_keypoint_list.size(); i++) {
            final int I = i;
            double distance_min = Double.MAX_VALUE;
            for (int j = 0; j < bsk_2_keypoint_list.size(); j++) {
                double distance = Keypoint.getHistogramDistance(bsk_1_keypoint_list.get(i), bsk_2_keypoint_list.get(j));
                if (distance < distance_min) {
                    distance_min = distance;
                }
            }
            distance_tab[I] = distance_min;
        }
        double distance_cumul_1 = 0;
        for (double distance : distance_tab) {
            distance_cumul_1 += distance;
        }
        distance_cumul_1 /= distance_tab.length;

        // compare with the mirrored
        for (int i = 0; i < distance_tab.length; i++) {
            distance_tab[i] = 0;
        }
        for (int i = 0; i < bsk_1_keypoint_list.size(); i++) {
            final int I = i;
            double distance_min = Double.MAX_VALUE;
            for (int j = 0; j < bsk_2_keypoint_list.size(); j++) {
                double distance = Keypoint.getHistogramDistance(bsk_1_keypoint_list.get(i), bsk_2_keypoint_list.get(j).getMirroredKeypoint());
                if (distance < distance_min) {
                    distance_min = distance;
                }
            }
            distance_tab[I] = distance_min;
        }
        double distance_cumul_2 = 0;
        for (double distance : distance_tab) {
            distance_cumul_2 += distance;
        }
        distance_cumul_2 /= distance_tab.length;

        if (distance_cumul_1 < distance_cumul_2) {
            return 1 / (distance_cumul_1 + 1);
        } else {
            return 1 / (distance_cumul_2 + 1);
        }

    }

    public static double matchSymmetricMirrorInvariant(bsk bsk_1, bsk bsk_2) {

        double value = (matchMirrorInvariant(bsk_1, bsk_2) + matchMirrorInvariant(bsk_2, bsk_1)) / 2;

        return value;
    }

    public void serialize(String file_name) {
        try {
            FileOutputStream fileOut = new FileOutputStream(file_name);
            ObjectOutputStream out = new ObjectOutputStream(fileOut);
            out.writeObject(this);
            out.close();
            fileOut.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static bsk deserialize(String file_name) {
        try {
            FileInputStream fileIn = new FileInputStream(file_name);
            ObjectInputStream in = new ObjectInputStream(fileIn);
            bsk cci = (bsk) (in.readObject());
            in.close();
            fileIn.close();
            return cci;
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }
    }

    public static void main(String[] args) {

//        args = new String[1];
//        args[0] = "alpha1.png";
//////        
////        IplImage image = cvLoadImage(args[0]);
//        IplImage image_big = cvLoadImage(args[0]);
//        int size_factor = 2;
//        CvSize size = cvSize(image_big.width() / size_factor, image_big.height() / size_factor);
//        IplImage image = IplImage.create(size, image_big.depth(), image_big.nChannels());
//        cvResize(image_big, image);
//////        IplImage image_big = cvLoadImage(args[0]);
//////        int size_factor = 1;
//////        CvSize size = cvSize(image_big.width() / size_factor, image_big.height() / size_factor);
//////        IplImage image = IplImage.create(size, image_big.depth(), image_big.nChannels());
//////        cvResize(image_big, image);
////
//        DistanceTransform.K = 11;
//        bsk bsk = new bsk(image);
////        bsk.performScaleSpacebsk();
//        bsk.performbsk();
//        bsk.showStableKeypoints(0);
//
//        String ser_name = args[0].substring(0, args[0].indexOf(".png")) + ".ser";
////        System.out.println("ser_name = " + ser_name);
//        bsk.serialize(ser_name);
//         
//        cvReleaseImage(image);

//        bsk bsk_1 = bsk.deserialize("3.ser");
//        bsk bsk_2 = bsk.deserialize("3.ser");
////        bsk.STABILITY_THRESHOLD = 0.5;
//        double sim = bsk.match(bsk_1, bsk_2);
//        System.out.println("sim = " + sim);





        // to prepare datasets

//        args = new String[5];
//        args[0] = "camel-1.pgm";
//        args[1] = "1";
//        args[2] = "4";
//        args[3] = "8";
//        args[4] = "11";

        // UNCOMMENT THIS FOR ZANIBBI AND TOBACCO
        IplImage image_big = cvLoadImage(args[0]);
        image_big = Utilities.binarize(image_big);
        int SIZE_FACTOR = Integer.valueOf(args[1]);
        CvSize size = cvSize(image_big.width() / SIZE_FACTOR, image_big.height() / SIZE_FACTOR);
        IplImage image = IplImage.create(size, image_big.depth(), image_big.nChannels());
        cvResize(image_big, image);
        cvReleaseImage(image_big);
        image = Utilities.binarize(image);

        Keypoint.DISTANCE_PARAMETER = Integer.valueOf(args[2]);
        Keypoint.ANGLE_PARAMETER = Integer.valueOf(args[3]);
        DistanceTransform.K = Integer.valueOf(args[4]);

//        DistanceTransform.K = 3;
//        bsk.N = 30;
        bsk bsk = new bsk(image);
        bsk.performbsk();
        bsk.performRegression();

//        bsk.N = 298;//bsk.regression_index;
//        System.out.println("bsk.regression_index = " + bsk.regression_index);
//        System.out.println("bsk.N = " + bsk.N);
//        int n = bsk.keypointList.size();    
//        System.out.println("Original number of keypoints = " + n);
//        bsk.calculateSparcity();
//        System.out.println("Showing " + bsk.N + " keypoints..");
//        bsk.showKeypoints();
//        bsk.showKeypointDescendingScore();


        // show keypoint scores curve
        // draw curve
//        double x_array[] = new double[bsk.keypointList.size()];
//        double y_array[] = new double[bsk.keypointList.size()];
//        double score_cumul = 0;
//        try {
//            File file = new File("data_gnuplot.txt");
//            BufferedWriter writer = new BufferedWriter(new FileWriter(file));
//            for (int index = 1; index < bsk.keypointList.size(); index++) {
//
//                String str = index + " " + Math.log(1 + score_cumul);
//
//                x_array[index] = index + 1;
//                y_array[index] = Math.log(1 + score_cumul);
//
//                score_cumul += bsk.keypointList.get(index).score;
////                String str = (index + 1) + " " + bsk.keypointList.get(index).score;
//                writer.write(str);
//                writer.newLine();
//                writer.flush();
//            }
//            writer.close();
////            SegmentedRegression regress = new SegmentedRegression(x_array, y_array);
//////            regress.performTwoSegmentsRegression();
////            regress.performThreeSegmentsRegression();
////            int regression_index = regress.regression_x_index;
////            System.out.println("regression_index = " + regression_index);
//            Runtime.getRuntime().exec("gnuplot plot.plt").waitFor();
//            IplImage plot = cvLoadImage("plot.png");
//            cvShowImage("window", plot);
//            cvWaitKey(0);
//        } catch (Exception e) {
//            e.printStackTrace();
//        }



        String ser_name = args[0].substring(0, args[0].indexOf(".")) + ".ser";
        bsk.serialize(ser_name);

        cvReleaseImage(image);
        cvReleaseImage(bsk.image_normalized);


//        bsk.N = 100;
//        bsk bsk_1 = bsk.deserialize("H1.ser");
//        bsk bsk_2 = bsk.deserialize("h1.ser");
//        double sim = bsk.matchSymmetric(bsk_1, bsk_2);
//        System.out.println("sim = " + sim);
    }
}
