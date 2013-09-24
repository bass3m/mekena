(ns mekena.linear
  (:require [incanter.core :as incore]
            [incanter.io :as io]
            [incanter.charts :as charts]))

;; Stanford Machine Learning using Clojure and Incanter
;; machine learning hw1: linear regression
;; config params
(def alpha 0.01)
(def iterations 1500)

(def mtrx (io/read-dataset "./resources/ex1data1.txt"))
;; select the x and y columns of the matrix
(def xs (incore/sel mtrx :cols 0))
(def ys (incore/sel mtrx :cols 1))

(incore/view (charts/scatter-plot
              :col0
              :col1
              :data mtrx
              :x-label "Population of City in 10,000s"
              :y-label "Profit in $10,000s"))

(defn h-ftn
  "The hypothesis function : h-theta(x) = theta0 + (theta1 * x)
  we add a column of 1 to x to produce a mx2 matrix.
  x and theta are matrices: x is mx2, theta is 2x1"
  [t0 t1 x]
  (->> x
       (map (partial * t1))
       (map (partial + t0))))

;; TODO eliminate repetitions
(defn calc-theta0
  [x y theta]
  (let [t0 (first theta)
        t1 (second theta)]
    (as-> x _
      (h-ftn t0 t1 _)
      (map - _ y)
      (reduce + _)
      (/ _ (count x))
      (* alpha _))))

(defn calc-theta1
  [x y theta]
  (let [t0 (first theta)
        t1 (second theta)]
    (as-> x _
      (h-ftn t0 t1 _)
      (map - _ y)
      (map * _ x)
      (reduce + _)
      (/ _ (count x))
      (* alpha _))))

(defn calc-theta
  [x y theta]
  (map - theta ((juxt (partial calc-theta0 x y) (partial calc-theta1 x y)) theta)))

(defn gradient-descent-history
  "Get theta values for last 10 iterations, for debugging really"
  [x y]
  (drop (- iterations 10) (take iterations (iterate (partial calc-theta x y) [0 1]))))

(defn gradient-descent
  "Calculate gradient descent, returns intercept: theta-0 and slope: theta-1"
  [x y]
  (drop (dec iterations) (take iterations (iterate (partial calc-theta x y) [0 1]))))

(defn predict-value
  [x xs ys]
  (as-> xs _
    (gradient-descent _ ys)
    (flatten _)
    (+ (first _) (* x (second _)))))

(defn compute-mse
  "Calculate the cost (mse : mean square error) function: 1/2m * Sigma(from 1,m) (Theta^T*X - Y)^2"
  [x y theta]
  (let [t0 (first theta)
        t1 (second theta)]
    (as-> x _
      (h-ftn t0 t1 _)
      (map - _ y)
      (map (fn [n] (* n n)) _)
      (reduce + _)
      (/ _ (* 2 (count x))))))

(defn calc-mse-hist
  "Verify that mean error square to decreasing"
  [x y]
  (->> y
       (gradient-descent-history x)
       (map (partial compute-mse x y))))

;; now plot the regression
(defn plot-linear-reg
  [x y]
  (let [view (charts/scatter-plot
               :col0
               :col1
               :data mtrx
               :x-label "Population of City in 10,000s"
               :y-label "Profit in $10,000s")
        slope-incpt (first (gradient-descent x y))
        intercept (first slope-incpt)
        slope (second slope-incpt)]
    (incore/view (charts/add-function
                  view
                  (fn [x] (+ intercept (* x slope)))
                  (apply min x)
                  (apply max x)))))
