*use "C:\Users\Franco\GitHub\teacher-predictions\data\PAA-evdoc.dta", clear
use "D:\Google Drive\Data_projects\teachers_pjepred\PAA-evdoc.dta", clear

preserve 
use "D:\Google Drive\Data_projects\teachers_pjepred\score_data_20180410", clear
sort rut nem
bys rut: gen dup = _n
keep if dup == 1
keep rut gpa nem
tempfile aux
save `aux' 
restore

merge m:1 rut using `aux', gen(_m) keep(match)
drop _m
cap drop rut
outsheet * using "C:\Users\Franco\GitHub\teacher-predictions\data\PAA-evdoc.csv" , comma nolabel replace

