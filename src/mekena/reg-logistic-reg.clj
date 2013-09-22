(ns mekena.reg-logistic-reg
  (:require [incanter.core :as i]
            [incanter.io :as io]
            [incanter.stats :as stats]
            [incanter.charts :as charts]))


;; Regularized logistic regression

;; read data
(def qa-data (io/read-dataset "./resources/ex2data2.txt"))

(def test1 (i/sel qa-data :cols 0))
(def test2 (i/sel qa-data :cols 1))
(def accepted (i/sel qa-data :cols 2))

;; plot the qa data for the exams, use group-by to separate
;; accepted vs. not accepted
(i/view (charts/scatter-plot
          :col0
          :col1
          :group-by :col2
          :data qa-data
          :x-label "Microchip test 1"
          :y-label "Microchip test 2"))

(def lambda 1)

(defn hypothesis
  "sigmoid function: 1/(1+e^-z)"
  [z]
  (/ 1 (inc (Math/exp (- z)))))

(defn map-features
  "Return vector containing powers that features will be used to generate
  the regularization parameter. map the features into all polynomial terms
  of x1 and x2 up to the degree'th power. Assumes only 2 features (for now)"
  [degree]
  (for [i (range 1 (inc degree)) j (range (inc i))] [(- i j) j]))

(defn regularize-features [x1 x2 poly-vec]
  (map (fn [x1i x2i]
         (reduce (fn [acc [x1-term x2-term]]
                       (conj acc (i/mult (i/pow x1i x1-term)
                                             (i/pow x2i x2-term))))
                     [] poly-vec)) x1 x2))

(defn regularize
  [xs]
  (let [x1 (i/sel qa-data :cols 0)
        x2 (i/sel qa-data :cols 1)
        polynomial-terms (map-features 6)
        regularized (regularize-features x1 x2 polynomial-terms)]
    (-> x1
        i/nrow
        (repeat 1)
        (i/bind-columns regularized))))

; X : 118x28 theta is 28x1
; test: (compute-cost (regularize qa-data) accepted (repeat 28 0))
; 6.93e-01
(defn compute-cost
  "Calculate the cost of using theta (mse : mean square error):
   J(θ) = −1/m(log(g(Xθ))^T.y + (log(1−g(Xθ)))^T.(1−y))
   -1/m Sigma (1,m)[ y.log( h(theta.x) ) + (1 - y).log( 1 - h(theta.x) )]
   input xs is dataset "
  [xs y theta]
  (let [theta-x      (i/mmult xs theta)
        h-tx         (i/matrix-map hypothesis theta-x)
        y-1-term     (i/mmult (i/trans (i/log h-tx)) y)
        y-0-term     (i/mmult (i/trans (i/log (i/minus 1 h-tx))) (i/minus 1 y))
        m            (count y)]
    (-> y-1-term
        (i/plus y-0-term)
        (i/div (- m)))))

(defn regularized-cost
  "Compute the regularized cost, which adds the regularized parameter to the
  logistic regression cost: adds λ/2m .Sigma j=(1,n) θj^2"
  [xs y theta]
  (let [cost (compute-cost xs y theta)]
    (-> theta
        ((juxt i/trans identity))
        ((partial apply i/mmult))
        (i/mult lambda)
        (i/div (* 2 (count theta)))
        (i/plus cost))))

(defn calc-theta
  "θ = θ − α/m.X^T.(g(Xθ)−y⃗ ))
  Gradient descent is: theta - (alpha/m).x^T(g(x.theta) - y)
  calculate an iteration"
  [xs y theta]
  (let [xs' (i/trans xs)]
    (->> theta
         (i/mmult xs)
         (i/matrix-map hypothesis)
         (i/minus y)
         (i/mmult xs')
         (i/mult alpha (/ -1 (count xs))))))

(defn calc-next-theta
  "calculate the next theta value"
  [xs y theta]
  (->> theta
       (calc-theta xs y)
       (i/minus theta)))

