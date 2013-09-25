(defproject mekena "0.1.0-SNAPSHOT"
  :description "ML fun"
  :url "http://example.com/FIXME"
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :dependencies [[org.clojure/clojure "1.5.1"]
                 [incanter "1.5.4"]]
  :repl-options {:prompt
                 (fn [ns]
                   (str "\u001B[35m[\u001B[34m"
                        ns "\u001B[35m]\u001B[33mclj-λ)\u001B[m "))
                 :welcome (println "Welcome to Clojure!")}
  :min-lein-version "2.0.0"
  :jvm-opts [;;"-Djava.awt.headless=true"
             "-Xms256m" "-Xmx1024m"])
