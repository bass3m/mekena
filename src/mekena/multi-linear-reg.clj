(ns mekena.multi-linear-reg
  (:require [incanter.core :as incore]
            [incanter.io :as io]
            [incanter.stats :as stats]
            [incanter.charts :as charts]))

;; Stanford Machine Learning using Clojure and Incanter
;; machine learning multi-variate linear regression
;; config params
(def alpha 0.01)
(def iterations 1500)

;; read housing data
(def mtrx (io/read-dataset "./resources/ex1data2.txt"))
;; select the x and y columns of the matrix
(def square-footage (incore/sel mtrx :cols 0))
(def bedrooms (incore/sel mtrx :cols 1))
(def price (incore/sel mtrx :cols 2))

(incore/view (charts/scatter-plot
              :col0
              :col1
              :data mtrx
              :x-label "Population of City in 10,000s"
              :y-label "Profit in $10,000s"))

(defn h-ftn
  "The hypothesis function : X . theta
  h-theta(x) = (theta0 * 1) + (theta1 * x1) + (theta2 * x2) ..
  we add a column of 1 to x to produce a [m x (n+1)] matrix.
  theta [(n+1) x 1], produces a matrix of [m x 1]"
  [x theta]
  (incore/mmult x theta))

(defn prepare-x
  "append a column of 1s as x0. Used as multiplier for theta0. Keep all cols except last"
  [mtrx]
  (-> mtrx
      incore/nrow
      (repeat 1)
      vec
      (incore/bind-columns (incore/to-matrix mtrx))
      (incore/sel :except-cols (incore/ncol mtrx))))

(defn calc-new-theta
  [x y theta]
  (let [x' (incore/trans x)]
    (-> x
        (incore/mmult theta)
        (incore/minus y)
        ((partial incore/mmult x'))
        (incore/div (count x))
        (incore/mult alpha)
        ((partial incore/minus theta)))))

;(defn calc-theta
  ;[x y theta]
  ;(map - theta ((juxt (partial calc-theta0 x y) (partial calc-theta1 x y)) theta)))

(defn gradient-descent-history
  "Get theta values for last 10 iterations, for debugging really"
  [x y]
  (drop (- iterations 10) (take iterations (iterate (partial calc-theta x y) [0 1]))))

;; need to group the xs together or just return all butlast
(defn gradient-descent
  "Calculate gradient descent."
  ([mtrx] (gradient-descent mtrx iterations))
  ([mtrx iter]
   (let [x (prepare-x mtrx)
         y (incore/sel mtrx :cols (count mtrx))]
     (drop (dec iter)
           (take iter
                 (iterate (partial calc-new-theta x y)
                          (repeat (inc (count mtrx)) 1)))))))

;(defn predict-value
  ;[x xs ys]
  ;(as-> xs _
    ;(gradient-descent _ ys)
    ;(flatten _)
    ;(+ (first _) (* x (second _)))))

;(defn compute-mse
  ;"Calculate the cost (mse : mean square error) function: 1/2m * Sigma(from 1,m) (Theta^T*X - Y)^2"
  ;[x y theta]
  ;(let [t0 (first theta)
        ;t1 (second theta)]
    ;(as-> x _
      ;(h-ftn t0 t1 _)
      ;(map - _ y)
      ;(map (fn [n] (* n n)) _)
      ;(reduce + _)
      ;(/ _ (* 2 (count x))))))

;(defn calc-mse-hist
  ;"Verify that mean error square to decreasing"
  ;[x y]
  ;(->> y
       ;(gradient-descent-history x)
       ;(map (partial compute-mse x y))))

;;; now plot the regression
;(defn plot-linear-reg
  ;[x y]
  ;(let [view (charts/scatter-plot
               ;:col0
               ;:col1
               ;:data mtrx
               ;:x-label "Population of City in 10,000s"
               ;:y-label "Profit in $10,000s")
        ;slope-incpt (first (gradient-descent x y))
        ;intercept (first slope-incpt)
        ;slope (second slope-incpt)]
    ;(incore/view (charts/add-function
                  ;view
                  ;(fn [x] (+ intercept (* x slope)))
                  ;(apply min x)
                  ;(apply max x)))))
