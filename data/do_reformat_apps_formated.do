*use "D:\Dropbox\CDE_light\apps_formatted_20180410_mask.dta", clear
use "D:\Google Drive\Bases Franco\Chile\apps_formatted_20180410.dta", clear

keep rut proceso sit nem paa_* pce_* region male privado tFLcode_app
keep if sit == 24
sort rut proceso
bys rut: keep if _n == _N
duplicates report rut

gen fl_teacher = (tFLcode_app >= 73 & tFLcode_app <= 83)


/*
forvalues i = 2003/2010{
import excel using "D:\Google Drive\Bases Franco\Chile\censo_docentes_chile\docentes_`i'.xlsx", first clear	
save "D:\Google Drive\Bases Franco\Chile\censo_docentes_chile\docentes`i'run.dta", replace
}
*/

preserve
clear
forvalues i = 2003/2017{
append using "D:\Google Drive\Bases Franco\Chile\censo_docentes_chile\docentes`i'run", force
}
drop rut
duplicates drop doc_run, force
ren doc_run rut
tempfile is_teacher
save `is_teacher'
restore

merge 1:1 rut using `is_teacher', gen(_m) keep(master match) keepusing(ido_estado)
gen isteacher = (_m == 3)
drop ido_estado _m
drop rut

outsheet * using "C:\Users\Franco\GitHub\teacher-predictions\data\proceso_evdoc.csv" , comma nolabel replace
