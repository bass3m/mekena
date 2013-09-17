(ns mekena.logistic-reg
   (:require [incanter.core :as i]))

(defn sigmoid
  "sigmoid function: 1/(1+e^-z)"
  [z]
  (/ 1 (inc (i/exp (i/minus z)))))

