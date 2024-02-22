#|

Spacing effect

7/11: add *scale* parameter for time intervals, defaulting to 1

8/4: add predictions for n reinforcements

|#

(defun retrieve (activation &key (threshold 0.0) (noise 0.25))
  (let ((exp-act (exp (/ activation noise)))
        (exp-the (exp (/ threshold noise))))
    (/ exp-act (+ exp-act exp-the))))

(defun activation (refs &key (decay 0.5))
  (let ((act 0.0))
    (dolist (ref refs)
      (let ((lag (if (numberp ref) ref (first ref)))
            (weight (if (numberp ref) 1.0 (second ref))))
        (incf act (* weight (expt lag (- decay))))))
    (log act)))

(defun full-references (isi ri &key (decay 0.5) (delta 1.0))
  (declare (ignore decay delta))
  (list (list ri 1.0)
        (list (+ isi ri) 1.0)))

(defun even-references (isi ri &key (decay 0.5) (delta 1.0))
  (let* ((previous (exp (activation (list (list (+ isi delta) 1.0)) :decay decay)))
         (current (exp (activation (list (list delta 1.0)) :decay decay)))
         (weight (/ (- current previous) current)))
    (list (list ri weight)
          (list (+ isi ri) 1.0))))

(defun split-references (isi ri split &key (decay 0.5) (delta 1.0))
  (let* ((current (exp (activation (list (list delta 1.0)) :decay decay)))
         (split (exp (activation (list (list (+ (* split isi) delta) 1.0)) :decay decay)))
         (split-weight (/ (- current split) current))
         (end (exp (activation (list (list (+ isi delta) 1.0)
                                     (list (+ (* (- 1.0 split) isi) delta) split-weight)) :decay decay)))
         (end-weight (/ (- current end) current)))
    (list (list ri end-weight)
          (list (+ (* (- 1.0 split) isi) ri) split-weight)
          (list (+ isi ri) 1.0))))

(defun n-references (n isi ri &key (decay 0.5) (delta 1.0))
  (let ((baseline (exp (activation (list (list delta 1.0)) :decay decay)))
        (references (list (list delta 1.0))))
    (dotimes (i (1- n))
      (dolist (reference references)
        (incf (first reference) isi))
      (let* ((current (exp (activation references :decay decay)))
             (weight (/ (- baseline current) baseline)))
        (push (list delta weight) references)))
    (dolist (reference references)
      (incf (first reference) (- ri delta)))
    references))
    
(defparameter *scale* 24)

(defun experiment (&key (isis '(1 7 14 21 35 70 105)) (ris '(7 35 70 350))
                        (references 'full-references) (delta 1.0)
                        (decay 0.5) (threshold -1.0) (noise 0.25))
  (format t "ISI")
  (dolist (ri ris)
    (format t "~CRI=~3D" #\tab ri))
  (format t "~%")
  (dolist (isi isis)
    (format t "~D" isi)
    (dolist (ri ris)
      (format t "~C~6,3F" #\tab (* 100.0
                                   (retrieve
                                    (activation (funcall references (* *scale* isi) (* *scale* ri)
                                                         :decay decay :delta delta)
                                                :decay decay)
                                    :threshold threshold :noise noise))))
    (format t "~%")))

(defun split-experiment (&key (isis '(1 7 14 21 35 70 105)) (ris '(7 35 70 350)) (splits '(0.25 0.5 0.75))
                              (references 'split-references) (delta 1.0)
                              (decay 0.5) (threshold -1.0) (noise 0.25))
  (dolist (split splits)
    (format t "SPLIT=~4,2F~%" split)
    (format t "ISI")
    (dolist (ri ris)
      (format t "~CRI=~3D" #\tab ri))
    (format t "~%")
    (dolist (isi isis)
      (format t "~D" isi)
      (dolist (ri ris)
        (format t "~C~6,3F" #\tab (* 100.0
                                     (retrieve
                                      (activation (funcall references (* *scale* isi) (* *scale* ri) split
                                                           :decay decay :delta delta)
                                                  :decay decay)
                                      :threshold threshold :noise noise))))
      (format t "~%"))))

#|
? (experiment :threshold -1.5 :noise 0.5 :references 'full-references)
ISI	RI=  7	RI= 35	RI= 70	RI=350
1	91.488	69.358	53.263	18.648
7	89.318	67.740	52.266	18.520
14	87.714	66.145	51.218	18.375
21	86.588	64.788	50.273	18.234
35	85.053	62.580	48.633	17.965
70	82.937	58.811	45.540	17.354
105	81.763	56.355	43.332	16.819
NIL
? (experiment :threshold -1.5 :noise 0.5 :references 'even-references)
ISI	RI=  7	RI= 35	RI= 70	RI=350
1	81.235	48.417	32.176	 8.735
7	84.018	58.253	42.346	13.323
14	83.314	59.104	43.997	14.548
21	82.612	58.811	44.270	15.065
35	81.561	57.659	43.852	15.485
70	80.058	54.976	41.994	15.593
105	79.226	53.039	40.347	15.384
NIL
? *scale*
24
? (split-experiment :threshold -3.0 :noise 0.5 :references 'split-references)
SPLIT=0.25
ISI	RI=  7	RI= 35	RI= 70	RI=350
1	91.914	70.797	55.007	19.757
7	91.963	75.729	62.232	25.688
14	90.621	75.081	62.390	26.654
21	89.506	74.031	61.837	26.951
35	87.811	71.922	60.356	27.018
70	85.174	67.668	56.800	26.453
105	83.567	64.570	53.925	25.700
SPLIT=0.50
ISI	RI=  7	RI= 35	RI= 70	RI=350
1	92.429	72.323	56.867	20.988
7	92.116	76.238	62.919	26.278
14	90.720	75.414	62.848	27.074
21	89.583	74.289	62.195	27.291
35	87.867	72.106	60.613	27.274
70	85.210	67.782	56.957	26.622
105	83.594	64.654	54.041	25.828
SPLIT=0.75
ISI	RI=  7	RI= 35	RI= 70	RI=350
1	92.657	73.018	57.729	21.581
7	92.182	76.461	63.222	26.542
14	90.764	75.560	63.050	27.260
21	89.617	74.401	62.352	27.442
35	87.892	72.187	60.725	27.387
70	85.226	67.831	57.026	26.696
105	83.607	64.691	54.091	25.884
NIL
? (dolist (reference (reverse (n-references 10 10 100)))
    (format t "~8,6F~%" (second reference)))
1.000000
0.698489
0.571180
0.495755
0.444257
0.406169
0.376502
0.352541
0.332656
0.315807
NIL
? (dolist (reference (reverse (n-references 10 100 1000)))
    (format t "~8,6F~%" (second reference)))
1.000000
0.900496
0.839863
0.795275
0.759786
0.730250
0.704944
0.682816
0.663168
0.645513
NIL
? (dolist (reference (reverse (n-references 10 25 250)))
    (format t "~8,6F~%" (second reference)))
1.000000
0.803884
0.702317
0.634990
0.585409
0.546638
0.515106
0.488736
0.466215
0.446665
NIL
? (dolist (reference (reverse (n-references 10 50 500)))
    (format t "~8,6F~%" (second reference)))
1.000000
0.859972
0.780076
0.723818
0.680507
0.645429
0.616063
0.590896
0.568945
0.549532
NIL
? (dolist (reference (reverse (n-references 10 5 50)))
    (format t "~8,6F~%" (second reference)))
1.000000
0.591752
0.456907
0.385049
0.338886
0.306080
0.281241
0.261596
0.245559
0.232147
NIL
? (dolist (reference (reverse (n-references 10 1000 10000)))
    (format t "~8,6F~%" (second reference)))
1.000000
0.968393
0.947037
0.930164
0.915942
0.903518
0.892412
0.882326
0.873057
0.864462
NIL
? (dolist (reference (reverse (n-references 10 500 5000)))
    (format t "~8,6F~%" (second reference)))
1.000000
0.955323
0.925712
0.902636
0.883401
0.866757
0.852005
0.838709
0.826577
0.815399
NIL
? (dolist (reference (reverse (n-references 10 250 2500)))
    (format t "~8,6F~%" (second reference)))
1.000000
0.936881
0.896188
0.865086
0.839563
0.817771
0.798681
0.781656
0.766268
0.752216
NIL
|#