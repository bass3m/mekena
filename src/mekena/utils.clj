(ns mekena.utils)

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

(defn prepare-features
  "append a column of 1s as feature 0"
  [xs]
  (-> xs
      i/nrow
      (repeat 1)
      (i/bind-columns xs)))

