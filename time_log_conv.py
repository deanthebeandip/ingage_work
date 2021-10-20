#convert log to analysis of delays

import datetime

filename = 'att_log_211014_dean_good.txt'
file1 = open(filename, 'r')
Lines = file1.readlines()
outLines = []

# Strips the newline character
for line in Lines:
    curr_line = "{}".format(line.strip())
    dl_time = curr_line[curr_line.find("Line")+22:curr_line.find("--ENTERING DOWNLOAD ONCE")-7]
    att_time = curr_line[curr_line.find("AttentionTime")+27:curr_line.find("ContainerName")-7]
    h_num = curr_line[curr_line.find("participantVideo_")+54:curr_line.find(".h264")]

    #find the time stamps n shit...
    dl_h = int(dl_time[0:2])
    dl_m = int(dl_time[3:5])
    dl_s = int(dl_time[6:8])
    att_h = int(att_time[0:2])
    att_m = int(att_time[3:5])
    att_s = int(att_time[6:8])
    h_diff = dl_h - att_h
    m_diff = dl_m - att_m
    s_diff = dl_s - att_s


    h_sign = 0
    if h_diff >= 0: h_sign = 1

    delay = 0
    if h_sign:
        if s_diff >= 0: #xx+
            delay += s_diff
            if m_diff >= 0: #+++
                delay += m_diff * 60 + h_diff * 60*60
            else: #+-+ case 3
                h_diff -= 1
                m_diff += 60
                delay += m_diff *60 + h_diff *60*60

        else: #sec is negative #xx-
            if m_diff > 0: #x+-
                m_diff -= 1
                delay += s_diff + 60 #give a minute to the seconds
                delay += m_diff *60
                delay += h_diff * 60*60
            else: #x--
                h_diff -= 1
                m_diff += 60
                s_diff += 60
                delay += s_diff + m_diff*60 + h_diff*60*60


    # outLines.append(att_time + " "+ dl_time + " delay: " + str(delay) + " " + h_num + "\n")
    outLines.append(h_num + "\t"+str(delay) + "\n")



out = filename.replace(".txt", "_out.txt")
file2 = open(out, 'w')
file2.writelines(outLines)
file2.close()

print("Finished converting file: " , filename)
