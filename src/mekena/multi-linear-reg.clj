(ns mekena.multi-linear-reg
  (:require [incanter.core :as incore]
            [incanter.io :as io]
            [incanter.stats :as stats]
            [incanter.charts :as charts]))

;; Stanford Machine Learning using Clojure and Incanter
;; machine learning multi-variate linear regression
;; config params
;; ofcourse we can just do this, but what's the fun with that:
;; (linear-model y (to-matrix (sel mtrx :cols [0 1])))
(def alpha 0.01)
(def iterations 1500)

;; read housing data
(def mtrx (io/read-dataset "./resources/ex1data2.txt"))
;; select the x and y columns of the matrix
(def square-footage (incore/sel mtrx :cols 0))
(def bedrooms (incore/sel mtrx :cols 1))
(def price (incore/sel mtrx :cols 2))

(defn calc-theta
  [feat y theta]
  (let [feat' (incore/trans feat)]
    (-> feat
        (incore/mmult theta)
        (incore/minus y)
        ((partial incore/mmult feat'))
        (incore/div (count feat))
        (incore/mult alpha)
        ((partial incore/minus theta)))))

(defn gradient-descent
  "Calculate gradient descent. Matrix contains multiple features with the y column
  as the last column (the dependent variable).
  An optional number of iterations can be passed as another parameter."
  [iter y feats]
  (drop (dec iter)
        (take iter
              (iterate (partial calc-theta feats y)
                       (repeat (incore/ncol feats) 1)))))

(defn prepare-features
  "append a column of 1s as feature 0"
  [mtrx]
  (-> mtrx
      incore/nrow
      (repeat 1)
      vec
      (incore/bind-columns mtrx)))

(defn normalize
  "normalize a feature by subtracting the mean and scale by the std dev"
  [mtrx]
  (let [mean (stats/mean mtrx)
        sd (stats/sd mtrx)]
    (vec (map (comp (fn [n] (/ n sd)) (fn [n] (- n mean))) mtrx))))

(defn linear-regression
  "Calculate gradient descent. Matrix contains multiple features with the y column
  as the last column (the dependent variable).
  An optional number of iterations can be passed as another parameter."
  ([mtrx] (linear-regression mtrx 1500))
  ([mtrx iters] (let [y (incore/sel mtrx :cols (dec (incore/ncol mtrx)))
                      feats (incore/sel mtrx :except-cols (dec (incore/ncol mtrx)))]
                  (->> (incore/ncol feats)
                       range
                       (map (fn [c] (normalize (incore/sel feats :cols c))))
                       matrix
                       trans
                       prepare-features
                       (gradient-descent iters y)))))

(defn predict-value
  "fvec is a vector containing features to be predicted, mtrx is features and y"
  [fvec mtrx]
  (let [y (incore/sel mtrx :cols (dec (incore/ncol mtrx)))
        feats (incore/sel mtrx :except-cols (dec (incore/ncol mtrx)))
        features-stats (->> (incore/ncol feats)
                            range
                            (map (comp (juxt stats/mean stats/sd)
                                       (fn [c] (incore/sel feats :cols c)))))
        ;; append 1 for x0 (for intercept)
        normalized-inputs (cons 1 (map (fn [fval [mean sd]] (/ (- fval mean) sd))
                                       fvec features-stats))]
    (as-> mtrx _
      (linear-regression _)
      (reduce
        (fn [acc [coef inp]] (+ acc (* coef inp))) 0
        (map vector
             (map (fn [c] (first (incore/sel _ :cols c))) (range (incore/nrow (first _))))
             normalized-inputs)))))

(defn compute-mse
  "Calculate the cost (mse : mean square error) function:
  1/2m * (X*Theta - Y)^T.(X*Theta - Y)"
  [x y theta]
  (let [normalized-x (->> (incore/ncol x)
                          range
                          (map (fn [c] (normalize (incore/sel x :cols c))))
                          matrix
                          trans
                          prepare-features)
        h-fn (incore/mmult normalized-x theta)]
    (as-> h-fn _
      (incore/minus _ y)
      ((juxt incore/trans identity) _)
      (incore/mmult (first _) (second _))
      (incore/div _ (* 2 (incore/nrow x))))))
