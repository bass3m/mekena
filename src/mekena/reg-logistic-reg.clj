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
        polynomial-terms (map-features 6)]
    (regularize-features x1 x2 polynomial-terms)))
