(ns mekena.logistic-reg
   (:require [incanter.core :as i]))

(defn sigmoid
  "sigmoid function: 1/(1+e^-z)
  Input can be a scalar, vector or matrix."
  [z]
  (i/div 1 (i/plus 1 (i/exp (i/minus z)))))

