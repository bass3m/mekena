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

(def alpha 1)
(def iterations 400)
(def lambda 1)

(defn hypothesis
  "sigmoid function: 1/(1+e^-z)"
  [z]
  (/ 1 (inc (Math/exp (- z)))))

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

(defn calc-regularized-theta
  [theta lambda m]
  (let [theta-size (count theta)
        ones (i/matrix 1 theta-size theta-size)
        iden (i/identity-matrix theta-size)]
    (->> iden
         (i/minus ones)
         first
         i/trans
         (i/mult theta lambda (/ 1 m)))))

(defn calc-regularized-theta
  [theta lambda m]
  (let [theta-size (count theta)
        ones (i/matrix theta-size theta-size)
        iden (i/identity-matrix theta-size)]
  (-> theta
      (assoc 0 0)
      (i/mult lambda (/ 1 m)))))

(defn calc-next-theta
  "calculate the next theta value"
  [xs y theta]
  (->> theta
       (calc-theta xs y)
       (i/plus (calc-regularized-theta theta lambda (count xs)))
       (i/minus theta)))

(defn gradient-descent
  "Calculate gradient descent. Matrix contains multiple features with the y column
  as the last column (the dependent variable).
  An optional number of iterations can be passed as another parameter."
  [iter y xs]
  (drop (dec iter)
        (take iter (iterate (partial calc-next-theta xs y)
                            (repeat (i/ncol xs) 0))))

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

(defn regularized-logistic-regression
  "Calculate gradient descent. input is Dataset containing multiple features
  with the y column as the last column (the dependent variable).
  An optional number of iterations can be passed as another parameter."
  ([dataset] (regularized-logistic-regression dataset iterations))
  ([dataset iters] (let [y (i/sel dataset :cols (dec (i/ncol dataset)))
                         xs (i/sel dataset :except-cols (dec (i/ncol dataset)))
                         reg-x (regularize xs)]
                     (first (gradient-descent iters y reg-x)))))

(defn predict
  "Predict the outcome (y values) given x and the calculated thetas"
  [xs theta]
  (->> theta
       (i/mmult xs)
       (i/matrix-map (comp (fn [y] (cond-> 0 (>= y 0.5) inc)) hypothesis))))

(defn prediction-accuracy
  [predicted actual]
  (let [accurate-count (->> actual
                            (map (fn [p a] (= p a)) predicted)
                            (filter true?)
                            count)]
    (println "Prediction accuracy:" (double (* 100
                                               (/ accurate-count
                                                 (count predicted)))) "%")))
