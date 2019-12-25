from selenium import webdriver
browser = webdriver.Safari(executable_path='/usr/bin/safaridriver')
browser.get('https://www.cnbc.com/site-map/')
yy = 0
mlst = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
dlst = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
for y in range(11):
    yy += 1
    mm = 0
    for m in range(12):
        mm += 1
        dd = 0
        for d in range(dlst[mm-1]):
            dd += 1
            browser.get('https://www.cnbc.com/site-map/%d/%s/%d/' % (2020-yy, mlst[mm-1], dd))
            articlelst = browser.find_elements_by_class_name('SiteMapArticleList-link')
            articlelst = [i.text for i in articlelst]
            with open('CNBC_TITLE.txt', 'a') as f:
                f.write(str(2020 - yy) + '/' + str(mm) + '/' + str(dd) + '\n')
                for i in articlelst:
                    f.write(i)
                f.write('\n===\n')
browser.quit()
