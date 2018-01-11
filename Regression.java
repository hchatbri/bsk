/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package bik;

import java.awt.Point;
import java.util.*;
import java.io.*;
import java.util.concurrent.*;
import java.util.Arrays;

/**
 *
 * @author houssem
 */
public class Regression {
    
    public List<Double> x_list;
    public List<Double> y_list;
    
    public double a;
    public double b;
    
    double min_x_regressed;
    
    public Regression(List<Double> x_list, List<Double> y_list) {
    this.x_list = x_list;
    this.y_list = y_list;

    min_x_regressed = 0;
}
    
    public void SimpleLinearRegression(double x[], double y[], int N) {
        if (N == 1) {
            a = 0;
            b = y[0];
        } else {
            double x_avg = 0;
            double y_avg = 0;
            for (int index = 0; index < N; ++index) {
                x_avg += x[index];
                y_avg += y[index];
            }
            x_avg /= N;
            y_avg /= N;

            double sum_1 = 0;
            double sum_2 = 0;

            for (int index = 0; index < N; ++index) {
                sum_1 += (x[index] - x_avg) * (y[index] - y_avg);
                sum_2 += (x[index] - x_avg) * (x[index] - x_avg);
            }

            a = sum_1 / sum_2;
            b = y_avg - (a) * x_avg;
        }
    }

    public double getAccurateXMin() {
        twoLinesRegressionUsingMinDistance();

        return min_x_regressed;
    }

    public void twoLinesRegressionUsingMinDistance() {

        int N = x_list.size();

        // first look for min of Y
        int index_y_min = 0;
        for (int index = 1; index < N; index++) {
            if (y_list.get(index) < y_list.get(index_y_min)) {
                index_y_min = index;
            }
        }

        double x[] = new double[N];
        double y[] = new double[N];

        for (int index = 0; index < N; ++index) {
            x[index] = x_list.get(index);
            y[index] = y_list.get(index);
        }

        SimpleLinearRegression(x, y, index_y_min + 1);
        double a_1 = a;
        double b_1 = b;
        //    SimpleLinearRegression(x, y, index_y_min, &a_1, &b_1);
        // calculate the cumulative distance between the first line and neighboring points
        double distance_1 = 0;
        for (int i = 0; i <= index_y_min; ++i) {
            Point p = new Point();
            p.setLocation(x[i], y[i]);
            distance_1 += pointToLineDistance(a_1, b_1, p);
        }

        // second line will have a slope = a_2 and an intercept equal to b_2
//        SimpleLinearRegression(x + index_y_min, y + index_y_min, N - index_y_min);
        SimpleLinearRegression(Arrays.copyOfRange(x, index_y_min, x.length), Arrays.copyOfRange(y, index_y_min, x.length), N - index_y_min);
        double a_2 = a;
        double b_2 = b;
        //    if (N - index_y_min >= 3) {
        //        SimpleLinearRegression(x + index_y_min, y + index_y_min, 3, &a_2, &b_2);    
        //    }
        //    else {
        //        SimpleLinearRegression(x + index_y_min, y + index_y_min, N - index_y_min, &a_2, &b_2);
        //    }
        // calculate the cumulative distance between the second line and neighboring points
        double distance_2 = 0;
        for (int i = index_y_min; i < N; ++i) {
            Point p = new Point();
            p.setLocation(x[i], y[i]);
            distance_2 += pointToLineDistance(a_2, b_2, p);
        }

        // get scale with the least distance to the intersection of the two lines

        // calculate the intersection of the two lines
        double intersection_x;
        double intersection_y;
        if (a_1 - a_2 != 0) {
            intersection_x = (b_2 - b_1) / (a_1 - a_2);
            intersection_y = a_1 * intersection_x + b_1;
        } else {
            intersection_x = x_list.get(index_y_min);
        }
        //    cout << endl << "intersection (" << intersection_x << ", " << intersection_y << ")" << endl;

        min_x_regressed = intersection_x;
    }

    public double pointToLineDistance(double a, double b, Point P) {
        
        Point Q_1 = new Point();
        Q_1.setLocation(P.x, a * P.x + b);
        
        Point Q_2 = new Point();
        Q_2.setLocation((P.y - b) / ((a == 0) ? 1 : a), P.y);

        double x_distance = Math.abs(Q_1.y - P.y);
        double y_distance = Math.abs(Q_2.x - P.x);

        if (x_distance < y_distance) {
            return x_distance;
        } else {
            return y_distance;
        }
    }

    public double getRegressedMinimum() {

        int N = x_list.size();

        // first look for min of Y
        int index_y_min = 0;
        for (int index = 1; index < N; index++) {
            if (y_list.get(index) < y_list.get(index_y_min)) {
                index_y_min = index;
            }
        }

        if (index_y_min == 0) {
            return 1;
        }
        else {
            if (index_y_min == N - 1) {
                return x_list.get(N - 1);
            }
            else {
                double y0 = y_list.get(index_y_min - 1);
                double y1 = y_list.get(index_y_min);
                double y2 = y_list.get(index_y_min + 1);

                double a0 = y0 / 8;
                double a1 = -y1 / 4;
                double a2 = y2 / 8;

                double a = a0 * (8);
                double b = -(a0 * (6) + a1 * (4) + a2 * (2));
                double c = a0 + a1 + a2;

                double minimum = -b / (2 * c);

                double real_min = minimum + x_list.get(index_y_min - 1);

                return real_min;
            }
        }
    }
}