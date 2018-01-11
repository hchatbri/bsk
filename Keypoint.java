/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package bik;

import bik.*;
import image_processing.Utilities;
import static org.bytedeco.javacpp.opencv_core.*;
import static org.bytedeco.javacpp.opencv_highgui.*;
import static org.bytedeco.javacpp.opencv_imgproc.*;

import java.awt.Point;
import java.util.*;

import java.io.*;

/**
 *
 * @author adapt
 */
public class Keypoint implements java.io.Serializable {

    public Point location;
    public Point nearestFgPoint;
    public int distance;
    public double stability = 0;
    public double score;
//    transient public IplImage image;
    transient public CvScalar color = cvScalar(Math.random() * 255, Math.random() * 255, Math.random() * 255, 255);//CvScalar.RED;
    public double histogram[][];
    public static int DISTANCE_PARAMETER = 4;
    public static int ANGLE_PARAMETER = 8;
//    public static final int DISTANCE_PARAMETER = 5;
//    public static final int ANGLE_PARAMETER = 12;

    public Keypoint(Point location) {//, IplImage image) {
        this.location = location;
//        this.image = image;
    }

    public void calculateScore2(List<Keypoint> keypointList) {

        double m = 0;
        double n = 1;

        for (Keypoint keypoint : keypointList) {
            if (Utilities.euclidean(location, keypoint.location) <= distance) {
                m += Math.abs(distance - keypoint.distance);

//                if (Math.abs(distance - keypoint.distance) <= 10) {
//                    n++;
//                }

                n++;
            }
        }
        score = distance / (m + 1);// / (1 + n);
//        score = distance;
    }

    public void calculateScore(List<Keypoint> keypointList) {

        double mean = 0;
        double sd = 0;

        for (int i = 0; i < DISTANCE_PARAMETER; i++) {
            for (int j = 0; j < ANGLE_PARAMETER; j++) {
                mean += histogram[i][j];
            }
        }
        mean /= DISTANCE_PARAMETER * ANGLE_PARAMETER;

        for (int i = 0; i < DISTANCE_PARAMETER; i++) {
            for (int j = 0; j < ANGLE_PARAMETER; j++) {
                sd += (histogram[i][j] - mean) * (histogram[i][j] - mean);
            }
        }
        sd /= DISTANCE_PARAMETER * ANGLE_PARAMETER;
        sd = Math.sqrt(sd);

        score = sd;

        double n = 1;

        for (Keypoint keypoint : keypointList) {
//            if (Utilities.euclidean(location, keypoint.location) <= distance) {
            if (Utilities.euclidean(location, keypoint.location) <= distance / 4) {
//            if (Utilities.euclidean(location, keypoint.location) <= distance / 2) {
                n++;
            }
        }
//        score = distance / (n + 1); !!!!!!! BUG !!!!!! MISSING +
        score = distance / (n + 1);
    }

//    public void calculateScoreDynamic(List<Keypoint> keypointList, double distance_mean) {
//
//        double mean = 0;
//        double sd = 0;
//
//        for (int i = 0; i < DISTANCE_PARAMETER; i++) {
//            for (int j = 0; j < ANGLE_PARAMETER; j++) {
//                mean += histogram[i][j];
//            }
//        }
//        mean /= DISTANCE_PARAMETER * ANGLE_PARAMETER;
//
//        for (int i = 0; i < DISTANCE_PARAMETER; i++) {
//            for (int j = 0; j < ANGLE_PARAMETER; j++) {
//                sd += (histogram[i][j] - mean) * (histogram[i][j] - mean);
//            }
//        }
//        sd /= DISTANCE_PARAMETER * ANGLE_PARAMETER;
//        sd = Math.sqrt(sd);
//
//        score = sd;
//
//        double n = 1;
//
//        for (Keypoint keypoint : keypointList) {
//            if (Utilities.euclidean(location, keypoint.location) <= distance_mean) {
//                n++;
//            }
//        }
//        score += distance / (n + 1);
//    }

    public void calculateFeatureVector(List<Point> pointList) {

        double radius = 1.5 * distance;
//        double radius = 1.25 * distance;

        histogram = new double[DISTANCE_PARAMETER][ANGLE_PARAMETER];
        for (int i = 0; i < DISTANCE_PARAMETER; i++) {
            for (int j = 0; j < ANGLE_PARAMETER; j++) {
                histogram[i][j] = (double) 0;
            }
        }

        for (Point q : pointList) {
            // calculate the norm
            double norm = Utilities.euclidean(location, q);
//            double norm_log = Math.log(1 + norm * 1);
            double norm_log = norm / radius;
            if (norm_log > 0 && norm_log < DISTANCE_PARAMETER) {
                // calculate the angle;
                int k = ANGLE_PARAMETER / 2;
                int angle_index = (int) (Utilities.angle(location, q) / (Math.PI / k));

                int norm_index = (int) Math.floor(norm_log);
                // update the histogram
                histogram[norm_index][angle_index]++;
            }
        }

        double sum = 1;
        for (int i = 0; i < DISTANCE_PARAMETER; i++) {
            for (int j = 0; j < ANGLE_PARAMETER; j++) {
                sum += histogram[i][j];
            }
        }

        for (int i = 0; i < DISTANCE_PARAMETER; i++) {
            for (int j = 0; j < ANGLE_PARAMETER; j++) {
                histogram[i][j] /= sum;
            }
        }

//        makeRotationInvariant();
    }

    public void makeRotationInvariant() {

        int k = ANGLE_PARAMETER / 2;

        int dominant_angle_index = (int) (Utilities.angle(location, nearestFgPoint) / (Math.PI / k));

        for (int angle_index = 0; angle_index < ANGLE_PARAMETER / 2; angle_index++) {
            for (int i = 0; i < DISTANCE_PARAMETER; i++) {

                double bin_original = histogram[i][angle_index];
                double bin_shifted = histogram[i][(dominant_angle_index + angle_index) % ANGLE_PARAMETER];

                histogram[i][angle_index] = bin_shifted;
                histogram[i][(dominant_angle_index + angle_index) % ANGLE_PARAMETER] = bin_original;
            }
        }
    }

    public Keypoint getMirroredKeypoint() {

        Keypoint mirrored_keypoint = new Keypoint(location);
        mirrored_keypoint.color = color;
        mirrored_keypoint.distance = distance;
        mirrored_keypoint.stability = stability;
        mirrored_keypoint.histogram = histogram;

        for (int angle_index = 0; angle_index < ANGLE_PARAMETER / 2; angle_index++) {
            for (int i = 0; i < DISTANCE_PARAMETER; i++) {

                double bin_original = histogram[i][angle_index];
                double bin_shifted = histogram[i][ANGLE_PARAMETER - 1 - angle_index];

                histogram[i][angle_index] = bin_shifted;
                histogram[i][ANGLE_PARAMETER - 1 - angle_index] = bin_original;
            }
        }

        return mirrored_keypoint;
    }

    public static double getHistogramDistance(Keypoint k_1, Keypoint k_2) {

        double hist1[][] = k_1.histogram;
        double hist2[][] = k_2.histogram;

        double distance = 0;
        for (int i = 0; i < DISTANCE_PARAMETER; i++) {
            for (int j = 0; j < ANGLE_PARAMETER; j++) {
                distance += (hist1[i][j] - hist2[i][j]) * (hist1[i][j] - hist2[i][j]) / (hist1[i][j] + hist2[i][j] + 1);
            }
        }
        distance /= 2;

        return distance;
    }
}
