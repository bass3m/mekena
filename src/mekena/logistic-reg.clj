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
   J(θ) =−1/m(log(g(Xθ))Ty+(log(1−g(Xθ)))T(1−y))
   -1/m Sigma (1,m)[ y.log( h(theta.x) ) + (1 - y).log( 1 - h(theta.x) )]
   input xs is dataset "
  [xs y theta]
  (let [;normalized-x (normalize-features xs)
        ;theta-x      (i/mmult normalized-x theta)
        theta-x      (i/mmult xs theta)
        h-tx         (i/matrix-map hypothesis theta-x)
        y-1-term     (i/mmult (i/trans (i/log h-tx)) y)
        y-0-term     (i/mmult (i/trans (i/log (i/minus 1 h-tx))) (i/minus 1 y))
        m            (count y)]
    (-> y-1-term
        (i/plus y-0-term)
        (i/div (- m)))))

;; deftest : compute-cost xs y [0 0 0] => 0.693
;; where: xs : (sel training-data :except-cols (dec (ncol training-data)))
;; y : (def y (sel training-data :cols (dec (ncol training-data))))

(def alpha 0.01)
(def iterations 400)

(defn calc-theta
  "Gradient descent is: theta - (alpha/m).x^T(g(x.theta) - y)
  calculate an iteration"
  [xs y theta]
  (let [xs' (i/trans xs)]
    (->> theta
         (i/mmult xs)
         (i/matrix-map hypothesis)
         (i/minus y)
         (i/mmult xs')
         (i/div (count xs))
         (i/mult alpha)
         (i/minus theta))))

(defn gradient-descent
  "Calculate gradient descent. Matrix contains multiple features with the y column
  as the last column (the dependent variable).
  An optional number of iterations can be passed as another parameter."
  [iter y xs]
  (drop (dec iter)
        (take iter (iterate (partial calc-theta xs y)
                            (repeat (i/ncol xs) 0)))))

(defn logistic-regression
  "Calculate gradient descent. input is Dataset containing multiple features
  with the y column as the last column (the dependent variable).
  An optional number of iterations can be passed as another parameter."
  ([dataset] (logistic-regression dataset iterations))
  ([dataset iters] (let [y (i/sel dataset :cols (dec (i/ncol dataset)))
                         xs (i/sel dataset :except-cols (dec (i/ncol dataset)))]
                     (->> (i/ncol xs)
                          range
                          (map (fn [c] (normalize (i/sel xs :cols c))))
                          matrix
                          trans
                          prepare-features
                          (gradient-descent iters y)))))
