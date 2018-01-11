package bik;

import java.awt.Point;
import java.util.*;
import java.io.*;
import java.util.concurrent.*;
import java.util.Arrays;

public class SegmentedRegression {

    // Data
    public double x_tab[];
    public double y_tab[];
    // Two segments' line coordinates
    public double a_1 = 0;
    public double b_1 = 0;
    public double a_2 = 0;
    public double b_2 = 0;
    // Break point's coordinates (the point belong to the curve and is the one minimizing the regression)
    public int regression_x_index = 0;
    
//    public int regression_x_index_1 = 0;
//    public int regression_x_index_2 = 0;

    public SegmentedRegression(double x_tab[], double y_tab[]) {
        this.x_tab = x_tab;
        this.y_tab = y_tab;
    }

    public double performTwoSegmentsRegression() {

        // p_first is the first point on the curve, p_last is the last one
        int N = x_tab.length;
        Point p_first = new Point();
        p_first.setLocation(x_tab[0], y_tab[0]);
        Point p_last = new Point();
        p_last.setLocation(x_tab[N - 1], y_tab[N - 1]);

        // p_middle takes all values between [0, N-1] and is used to estimate the break point minimizing the regression
        // check all middle points and detect the one minimizing the cumulative distance to the segments
        double distance_cumul_min = Double.MAX_EXPONENT;
        for (int i = 1; i < N - 2; i++) {
            
            double distance_cumul = 0;
            
            Point p_middle = new Point();
            p_middle.setLocation(x_tab[i], y_tab[i]);
            
            // calculate the cumulative distance for the points on the left
            for (int j = 1; j < i; j++) {
                Point p = new Point();
                p.setLocation(x_tab[j], y_tab[j]);
                distance_cumul += pointToLineDistance(p_first, p_middle, p);
            }
            // calculate the cumulative distance for the points on the right
            for (int j = i + 1; j < N - 2; j++) {
                Point p = new Point();
                p.setLocation(x_tab[j], y_tab[j]);
                distance_cumul += pointToLineDistance(p_middle, p_last, p);
            }
            
            if (distance_cumul < distance_cumul_min) {
                distance_cumul_min = distance_cumul;
                regression_x_index = i;
            }
        }
        
        // return the regression cost
        return distance_cumul_min;
    }

    public void performThreeSegmentsRegression() {

        // p_first is the first point on the curve, p_last is the last one
        int N = x_tab.length;
        Point p_first = new Point();
        p_first.setLocation(x_tab[0], y_tab[0]);
//        Point p_last = new Point();
//        p_last.setLocation(x_tab[N - 1], y_tab[N - 1]);

        int x_1 = -1;
        
        // p_middle takes all values between [0, N-1] and is used to estimate the break point minimizing the regression
        // check all middle points and detect the one minimizing the cumulative distance to the segments
        double distance_cumul_min = Double.MAX_EXPONENT;
        for (int i = 1; i < N - 2; i++) {
            
            double distance_cumul = 0;
            
            Point p_middle = new Point();
            p_middle.setLocation(x_tab[i], y_tab[i]);
            
            // calculate the cumulative distance for the points on the left
            for (int j = 1; j < i; j++) {
                Point p = new Point();
                p.setLocation(x_tab[j], y_tab[j]);
                distance_cumul += pointToLineDistance(p_first, p_middle, p);
            }
            
            // look for the point minimizing the regression to the right
            SegmentedRegression sr = new SegmentedRegression(Arrays.copyOfRange(x_tab, i, x_tab.length - 1), Arrays.copyOfRange(y_tab, i, y_tab.length - 1));
            distance_cumul += sr.performTwoSegmentsRegression();

            if (distance_cumul < distance_cumul_min) {
                distance_cumul_min = distance_cumul;
                regression_x_index = sr.regression_x_index + i;
                x_1 = i;
            }
        }
        
//        System.out.println("x_1 = " + x_1);
//        System.out.println("regression_x_index = " + regression_x_index);
    }
    
    // Distance between a point P and a line (A B)
    public double pointToLineDistance(Point A, Point B, Point P) {
        double normalLength = Math.sqrt((B.x - A.x) * (B.x - A.x) + (B.y - A.y) * (B.y - A.y));
        return Math.abs((P.x - A.x) * (B.y - A.y) - (P.y - A.y) * (B.x - A.x)) / normalLength;
    }
}
