(ns mekena.logistic-reg
  (:require [incanter.core :as i]
            [incanter.io :as io]
            [incanter.charts :as charts]))

;; read housing data
(def training-data (io/read-dataset "./resources/ex2data1.txt"))
;; select the x and y columns of the matrix
(def exam1 (i/sel training-data :cols 0))
(def exam2 (i/sel training-data :cols 1))
(def admitted (i/sel training-data :cols 2))

;; plot the training data for the exams, use group-by to separate
;; admitted vs. not admitted
(i/view (charts/scatter-plot
          :col0
          :col1
          :group-by :col2
          :data training-data
          :x-label "Exam 1 score"
          :y-label "Exam 2 score"))

(defn sigmoid
  "sigmoid function: 1/(1+e^-z)
  Input can be a scalar, vector or matrix."
  [z]
  (i/div 1 (i/plus 1 (i/exp (i/minus z)))))

(defn hypothesis
  "sigmoid function: 1/(1+e^-z)"
  [z]
  (/ 1 (inc (Math/exp (- z)))))

(defn prepare-features
  "append a column of 1s as feature 0"
  [xs]
  (-> xs
      i/nrow
      (repeat 1)
      (i/bind-columns xs)))

(defn normalize
  "normalize a feature by subtracting the mean and scale by the std dev"
  [mtrx]
  (let [mean (stats/mean mtrx)
        sd (stats/sd mtrx)]
    (vec (map (comp (fn [n] (/ n sd)) (fn [n] (- n mean))) mtrx))))

(defn normalize-features
  "Normalize all features of input dataset"
  [xs]
  (->> (i/ncol xs)
       range
       (map (fn [c] (normalize (i/sel xs :cols c))))
       matrix
       trans
       prepare-features))

(defn compute-cost
  "Calculate the cost of using theta (mse : mean square error):
   -1/m Sigma (1,m)[ y.log( h(theta.x) ) + (1 - y).log( 1 - h(theta.x) )]
   input xs is dataset "
  [xs y theta]
  (let [normalized-x (normalize-features xs)
        theta-x      (i/mmult normalized-x theta)
        h-tx         (i/matrix-map hypothesis theta-x)
        y-1-term     (i/mmult (i/trans y) (i/log h-tx))
        y-0-term     (i/mmult (i/trans (i/minus 1 y)) (i/log (i/minus 1 h-tx)))
        m            (count y)]
    (-> y-1-term
        (i/plus y-0-term)
        (i/div (- m)))))

;; deftest : compute-cost xs y [0 0 0] => 0.693
;; where: xs : (sel training-data :except-cols (dec (ncol training-data)))
;; y : (def y (sel training-data :cols (dec (ncol training-data))))

(defn gradient
  "Compute the partial derivatives and set gradient to the partial
  derivatives of the cost w.r.t. each parameter in theta
  gradient should have the same dimensions as theta"
  [theta x y])
