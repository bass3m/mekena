(ns mekena.logistic-reg
  (:require [incanter.core :as i]
            [incanter.io :as io]
            [incanter.stats :as stats]
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

;; XXX this doesn't work unless you are using trunk incanter with my fix
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
  [xs]
  (let [mean (stats/mean xs)
        sd (stats/sd xs)]
    (vec (map (comp (fn [n] (/ n sd)) (fn [n] (- n mean))) xs))))

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

;; deftest : compute-cost xs y [0 0 0] => 0.693
;; where: xs : (sel training-data :except-cols (dec (ncol training-data)))
;; y : (def y (sel training-data :cols (dec (ncol training-data))))
;; (compute-cost [[1 1 1 1][1 0 1 0][1 1 0 0][1 1 1 1]] [[0][1][1][0]] [1 0 0 -1])
;; 0.50320
;; (calc-theta [[1 1 1 1][1 0 1 0][1 1 0 0][1 1 1 1]] [[0][1][1][0]] [1 0 0 -1])
;; grad = 0.11553 0.18276 0.18276 0.25000

(def alpha 1)
(def iterations 400)

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

(defn gradient-descent
  "Calculate gradient descent. Matrix contains multiple features with the y column
  as the last column (the dependent variable).
  An optional number of iterations can be passed as another parameter."
  [iter y xs]
  (drop (dec iter)
        (take iter (iterate (partial calc-next-theta xs y)
                            (repeat (i/ncol xs) 0)))))


;; test that cost of theta at 400 iterations is 0.204
;; (compute-cost xn y (first (logistic-regression training-data 400)))
;; (logistic-regression training-data 400)
;; theta [1.66 3.88 3.62]
(defn logistic-regression
  "Calculate gradient descent. input is Dataset containing multiple features
  with the y column as the last column (the dependent variable).
  An optional number of iterations can be passed as another parameter."
  ([dataset] (logistic-regression dataset iterations))
  ([dataset iters] (let [y (i/sel dataset :cols (dec (i/ncol dataset)))
                         xs (i/sel dataset :except-cols (dec (i/ncol dataset)))
                         normalized-x (normalize-features xs)]
                     (gradient-descent iters y normalized-x))))

(defn plot-logistic-reg []
  (let [view (charts/scatter-plot
                :col0
                :col1
                :group-by :col2
                :data training-data
                :x-label "Exam 1 score"
                :y-label "Exam 2 score")
        thetas (first (logistic-regression training-data))
        slope (* (second thetas) (/ 1 (- (nth thetas 2))))
        xs (i/sel training-data :cols 1)
        min-x (apply min xs)
        max-x (apply max xs)
        intercept (+ max-x min-x)]
    (i/view (charts/add-function
              view
              ;; x2 = −1/θ2(θ1x1+θ0) when θX = 0
              ;; boundary occurs P(y=1 | X;θ)= hθ(x)=0.5 and sigmoid of 0.5
              ;; occurs as input is 0
              (fn [x] (+ intercept
                         (* (/ 1 (- (nth thetas 2)))
                            (+ (first thetas) (* x (second thetas))))))
              min-x max-x))))

(defn normalize-new-x
  "normalize a new feature value. Input is whole feature vector
  feature index (column) and new feature value"
  [xs n score]
  (let [x (i/sel xs :cols n)
        mean (stats/mean x)
        sd (stats/sd x)]
    ((comp (fn [exam-score] (/ exam-score sd))
           (fn [exam-score] (- exam-score mean))) score)))

(defn prediction-fn
  "Predict admittance for exam grades. Inputs: vector containing exam scores,
  vector containing theta values.
  make decision based on return of sigmoid ftn on theta.X > 0.5"
  [training-data scores]
  (let [thetas (first (logistic-regression training-data))
        xs (i/sel training-data :except-cols (dec (i/ncol training-data)))]
    (as-> scores _
          (map (partial normalize-new-x xs) (range) _)
          (matrix _)
          (trans _)
          (prepare-features _)
          (i/mmult _ thetas)
          (i/matrix-map hypothesis _)
          (first _))))

(defn admitted?
  "Are scores sufficient to get admitted ?"
  [training-data scores]
  (let [probability (prediction-fn training-data scores)]
    (if (>= probability 0.5)
      (println "Admitted: " probability)
      (println "Not Admitted: " probability))))
