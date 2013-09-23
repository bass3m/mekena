(ns mekena.utils
  (:require [incanter.core :as i]
            [incanter.io :as io]
            [incanter.stats :as stats]
            [incanter.charts :as charts]))

(defn rpartial
  "right currying"
  ([f & args]
     (fn [& prefix-args]
       (apply f (concat prefix-args args)))))

(defn normalize
  "normalize a feature by subtracting the mean and scale by the std dev"
  [mtrx]
  (let [mean (stats/mean mtrx)
        sd (stats/sd mtrx)]
    (vec (map (comp (fn [n] (/ n sd)) (fn [n] (- n mean))) mtrx))))

(defn hypothesis
  "sigmoid function: 1/(1+e^-z)"
  [z]
  (/ 1 (inc (Math/exp (- z)))))

(defn compute-cost
  "Calculate the cost of using theta (mse : mean square error):
   J(θ) = −1/m(log(g(Xθ))^T.y + (log(1−g(Xθ)))^T.(1−y))
   -1/m Sigma (1,m)[ y.log( h(theta.x) ) + (1 - y).log( 1 - h(theta.x) )]
   input xs is dataset "
  [xs y theta]
  (let [theta-x      (i/mmult xs theta)
        h-tx         (i/matrix-map utils/hypothesis theta-x)
        y-1-term     (i/mmult (i/trans (i/log h-tx)) y)
        y-0-term     (i/mmult (i/trans (i/log (i/minus 1 h-tx))) (i/minus 1 y))
        m            (count y)]
    (-> y-1-term
        (i/plus y-0-term)
        (i/div (- m)))))

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

(defn gradient-descent
  "Calculate gradient descent. Matrix contains multiple features with the y column
  as the last column (the dependent variable).
  An optional number of iterations can be passed as another parameter."
  [iter y xs]
  (drop (dec iter)
        (take iter (iterate (partial calc-next-theta xs y)
                            (repeat (i/ncol xs) 0)))))
