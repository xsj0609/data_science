-- 1. 查询"01"课程比"02"课程成绩高的学生的信息及课程分数
-- 	1.1 查询同时存在"01"课程和"02"课程的情况
-- 	1.2 查询存在"01"课程但可能不存在"02"课程的情况(不存在时显示为null)
-- 	1.3 查询不存在"01"课程但存在"02"课程的情况
-- 2. 查询平均成绩大于等于 60 分的同学的学生编号和学生姓名和平均成绩
-- 3. 查询在SC表存在成绩的学生信息
-- 4. 查询所有同学的学生编号、学生姓名、选课总数、所有课程的总成绩(没成绩的显示为 null )
-- 	4.1 查有成绩的学生信息
-- 5. 查询「李」姓老师的数量
-- 6. 查询学过「张三」老师授课的同学的信息
-- 7. 查询没有学全所有课程的同学的信息
-- 8. 查询至少有一门课与学号为"01"的同学所学相同的同学的信息
-- 9. 查询和"01"号的同学学习的课程完全相同的其他同学的信息
-- 10. 查询没学过「张三」老师讲授的任一门课程的学生姓名
-- 11. 查询两门及其以上不及格课程的同学的学号，姓名及其平均成绩
-- 12. 检索"01"课程分数小于 60，按分数降序排列的学生信息
-- 13. 按平均成绩从高到低显示所有学生的所有课程的成绩以及平均成绩
-- 14. 查询各科成绩最高分、最低分和平均分：
-- 	以如下形式显示：
-- 		课程ID，课程name，最高分，最低分，平均分，及格率，中等率，优良率，优秀率
-- 		及格为>=60，中等为：70-80，优良为：80-90，优秀为：>=90
-- 	输出：
-- 		课程号和选修人数，查询结果按人数降序排列，若人数相同，按课程号升序排列
-- 15. 按各科成绩进行排序，并显示排名， Score 重复时保留名次空缺
-- 	15.1 按各科成绩进行排序，并显示排名， Score 重复时合并名次
-- 16. 查询学生的总成绩，并进行排名，总分重复时保留名次空缺
-- 	16.1 查询学生的总成绩，并进行排名，总分重复时不保留名次空缺
-- 17. 统计各科成绩各分数段人数：课程编号，课程名称，[100-85]，[85-70]，[70-60]，[60-0] 及所占百分比
-- 18. 查询各科成绩前三名的记录
-- 19. 查询每门课程被选修的学生数
-- 20. 查询出只选修两门课程的学生学号和姓名
-- 21. 查询男生、女生人数
-- 22. 查询名字中含有「风」字的学生信息
-- 23. 查询同名同性学生名单，并统计同名人数
-- 24. 查询 1990 年出生的学生名单
-- 25. 查询每门课程的平均成绩，结果按平均成绩降序排列，平均成绩相同时，按课程编号升序排列
-- 26. 查询平均成绩大于等于 85 的所有学生的学号、姓名和平均成绩
-- 27. 查询课程名称为「数学」，且分数低于 60 的学生姓名和分数
-- 28. 查询所有学生的课程及分数情况（存在学生没成绩，没选课的情况）
-- 29. 查询任何一门课程成绩在 70 分以上的姓名、课程名称和分数
-- 30. 查询不及格的课程
-- 31. 查询课程编号为"01"且课程成绩在 80 分以上的学生的学号和姓名
-- 32. 求每门课程的学生人数
-- 33. 成绩不重复，查询选修「张三」老师所授课程的学生中，成绩最高的学生信息及其成绩
-- 34. 成绩有重复的情况下，查询选修「张三」老师所授课程的学生中，成绩最高的学生信息及其成绩
-- 35. 查询不同课程成绩相同的学生的学生编号、课程编号、学生成绩
-- 36. 查询每门功成绩最好的前两名
-- 37. 统计每门课程的学生选修人数（超过 5 人的课程才统计）。
-- 38. 检索至少选修两门课程的学生学号
-- 39. 查询选修了全部课程的学生信息
-- 40. 查询各学生的年龄，只按年份来算
-- 41. 按照出生日期来算，当前月日 < 出生年月的月日则，年龄减一
-- 42. 查询本周过生日的学生
-- 43. 查询下周过生日的学生
-- 44. 查询本月过生日的学生
-- 45. 查询下月过生日的学生

DROP TABLE IF EXISTS Student;
DROP TABLE IF EXISTS Course;
DROP TABLE IF EXISTS SC;
DROP TABLE IF EXISTS Teacher;

--学生表 Student
create table Student(Sid varchar(10),Sname nvarchar(10),Sage datetime,Ssex nvarchar(10));
insert into Student values('01' , '赵雷' , '1990-01-01' , '男');
insert into Student values('02' , '钱电' , '1990-12-21' , '男');
insert into Student values('03' , '孙风' , '1990-05-20' , '男');
insert into Student values('04' , '李云' , '1990-08-06' , '男');
insert into Student values('05' , '周梅' , '1991-12-01' , '女');
insert into Student values('06' , '吴兰' , '1992-03-01' , '女');
insert into Student values('07' , '郑竹' , '1989-07-01' , '女');
insert into Student values('08' , '王菊' , '1990-01-20' , '女');

--科目表 Course
create table Course(Cid varchar(10),Cname nvarchar(10),Tid varchar(10));
insert into Course values('01' , '语文' , '02');
insert into Course values('02' , '数学' , '01');
insert into Course values('03' , '英语' , '03');

--教师表 Teacher
create table Teacher(Tid varchar(10),Tname nvarchar(10));
insert into Teacher values('01' , '张三');
insert into Teacher values('02' , '李四');
insert into Teacher values('03' , '王五');

--成绩表 SC
create table SC(Sid varchar(10),Cid varchar(10),score decimal(18,1));
insert into SC values('01' , '01' , 80);
insert into SC values('01' , '02' , 90);
insert into SC values('01' , '03' , 99);
insert into SC values('02' , '01' , 70);
insert into SC values('02' , '02' , 60);
insert into SC values('02' , '03' , 80);
insert into SC values('03' , '01' , 80);
insert into SC values('03' , '02' , 80);
insert into SC values('03' , '03' , 80);
insert into SC values('04' , '01' , 50);
insert into SC values('04' , '02' , 30);
insert into SC values('04' , '03' , 20);
insert into SC values('05' , '01' , 76);
insert into SC values('05' , '02' , 87);
insert into SC values('06' , '01' , 31);
insert into SC values('06' , '03' , 34);
insert into SC values('07' , '02' , 89);
insert into SC values('07' , '03' , 98);
--------------------- 

select A.*,B.Cid,B.score from (select * from SC where Cid='01')A 
left join(select * from SC where Cid='02')B 
on A.Sid=B.Sid 
where A.score>B.score;
--1 查询“ 01 ”课程比" 02 "课程成绩高的学生的信息及课程分数
 
select * from (select * from SC where Cid='01')A 
left join (select * from SC where Cid='02')B on A.Sid=B.Sid
where B.Sid is not null;
--1.1 查询同时存在" 01 "课程和" 02 "课程的情况
 
select * from (select * from SC where Cid='01')A
left join (select * from SC where Cid='02')B on A.Sid=B.Sid;
--1.2 查询存在" 01 "课程但可能不存在" 02 "课程的情况(不存在时显示为null)
 
select * from SC where Cid='02'and Sid not in(select Sid from SC where Cid='01');
--1.3 查询不存在" 01 "课程但存在" 02 "课程的情况
 
select A.Sid,B.Sname,A.dc from(select Sid,AVG(score)dc from SC group by Sid)A
left join Student B on A.Sid=B.Sid where A.dc>=60;
--2. 查询平均成绩大于等于 60 分的同学的学生编号和学生姓名和平均成绩
 
select * from Student where Sid in (select distinct Sid from SC);
--3. 查询在 SC 表存在成绩的学生信息
 
select B.Sid,B.Sname,A.选课总数,A.总成绩 from
(select Sid,COUNT(Cid)选课总数,sum(score)总成绩 from SC group by Sid)A
right join Student B on A.Sid=B.Sid;
--4. 查询所有同学的学生编号、学生姓名、选课总数、所有课程的总成绩(没成绩的显示为null)
 
select A.Sid,B.Sname,A.选课总数,A.总成绩 from
(select Sid,COUNT(Cid)选课总数,sum(score)总成绩 from SC group by Sid)A
left join Student B on A.Sid=B.Sid;
--4.1 查有成绩的学生信息
 
select COUNT(*)李姓老师数量 from Teacher where Tname like '李%';
--5.查询「李」姓老师的数量 
 
select * from Student
where Sid in(select distinct Sid from SC 
where Cid=(select Cid from Course 
where Tid=(select Tid from Teacher where Tname='张三')));
--6.查询学过「张三」老师授课的同学的信息 
 
select * from Student where Sid in(select Sid from SC group by Sid having COUNT(Cid)<3);
--7.查询没有学全所有课程的同学的信息 
 
select * from Student 
where Sid in(select distinct Sid from SC where Cid in(select Cid from SC where Sid='01')
);
--8. 查询至少有一门课与学号为" 01 "的同学所学相同的同学的信息 
 
select * from Student 
where Sid in(select Sid from SC where Cid in(select distinct Cid from SC where Sid='01') and Sid<>'01'
group by Sid
having COUNT(Cid)>=3);
--9. 查询和" 01 "号的同学学习的课程完全相同的其他同学的信息 
 
select Sname from Student 
where Sid not in(select Sid from SC 
where Cid in(select Cid from Course where Tid in(select Tid from Teacher where Tname='张三')
)
);
--10. 查询没学过「张三」老师讲授的任一门课程的学生姓名 
 
select A.Sid,A.Sname,B.平均成绩 from Student A right join
(select Sid,AVG(score)平均成绩 from SC where score<60 group by Sid having COUNT(score)>=2)B
on A.Sid=B.Sid;
--11.查询两门及其以上不及格课程的同学的学号，姓名及其平均成绩 
 
select Sid,score from SC where Cid='01' and score<60 order by score desc;
--12.检索" 01 "课程分数小于 60 ，按分数降序排列的学生信息
 
select Sid,max(case Cid when '01' then score else 0 end)'01',
max(case Cid when '02' then score else 0 end)'02',
MAX(case Cid when '03' then score else 0 end)'03',AVG(score)平均分 from SC
group by Sid order by 平均分 desc;
--13. （静态写法）按平均成绩从高到低显示所有学生的所有课程的成绩以及平均成绩
 
SELECT
		DISTINCT A.Cid,Cname,max_score,min_score,avg_score,pass_rate--,中等率,优良率,优秀率 
FROM SC A

LEFT JOIN Course 
ON A.Cid=Course.Cid

LEFT JOIN 
		(SELECT Cid,
				MAX(score) max_score,
				MIN(score) min_score,
				ROUND(AVG(score),1) avg_score 
		FROM SC 
		GROUP BY Cid
		)B 
ON A.Cid=B.Cid

LEFT JOIN 
		(SELECT Cid,
				ROUND((SUM(CASE WHEN score>=60 THEN 1 ELSE 0 END)*1.00)/COUNT(*),3) pass_rate
		FROM SC 
		GROUP BY Cid
		)C 
ON A.Cid=C.Cid

left join (select Cid,
				round((sum(case when score >=70 and score<80 then 1 else 0 end)*1.00)/COUNT(*),3) mid_rate 
			from SC
			group by Cid
			)D 
on A.Cid=D.Cid
left join (select Cid,(convert(decimal(5,2),(sum(case when score >=80 and score<90 then 1 else 0 end)*1.00)/COUNT(*))*100)优良率 from SC group by Cid)E 
on A.Cid=E.Cid
left join (select Cid,(convert(decimal(5,2),(sum(case when score >=90 then 1 else 0 end)*1.00)/COUNT(*))*100)优秀率 
from SC group by Cid)F 
on A.Cid=F.Cid;
--14.查询各科成绩最高分、最低分和平均分：
	--以如下形式显示：课程 ID ，课程 name ，最高分，最低分，平均分，及格率，中等率，优良率，优秀率
	--及格为>=60，中等为：70-80，优良为：80-90，优秀为：>=90
 
select *,RANK()over(order by score desc)排名 from SC;
--15. 按各科成绩进行排序，并显示排名，Score 重复时保留名次空缺
 
select *,DENSE_RANK()over(order by score desc)排名 from SC;
--15.1 按各科成绩进行排序，并显示排名，Score 重复时合并名次
 
select *,RANK()over(order by 总成绩 desc)排名 from(
select Sid,SUM(score)总成绩 from SC group by Sid)A;
--16. 查询学生的总成绩，并进行排名，总分重复时保留名次空缺
 
select *,dense_rank()over(order by 总成绩 desc)排名 from(
select Sid,SUM(score)总成绩 from SC group by Sid)A;
--16.1 查询学生的总成绩，并进行排名，总分重复时不保留名次空缺
 
select distinct A.Cid,B.Cname,C.[100-85],C.所占百分比,D.[85-70],D.所占百分比,E.[70-60],E.所占百分比,F.[60-0],F.所占百分比
from SC A 
left join Course B ON A.Cid=B.Cid
left join (select Cid,sum(case when score>85 and score<=100 then 1 else null end)[100-85],
convert(decimal(5,2),(sum(case when score>85 and score<100 then 1 else null end))*1.00/COUNT(*))*100 所占百分比 from SC group by Cid)C on A.Cid=C.Cid
left join (select Cid,sum(case when score>70 and score<=85 then 1 else null end)[85-70],
convert(decimal(5,2),(sum(case when score>70 and score<=85 then 1 else null end))*1.00/COUNT(*))*100 所占百分比 from SC group by Cid)D on A.Cid=D.Cid
left join (select Cid,sum(case when score>60 and score<=70 then 1 else null end)[70-60],
convert(decimal(5,2),(sum(case when score>60 and score<=70 then 1 else null end))*1.00/COUNT(*))*100 所占百分比 from SC group by Cid)E on A.Cid=E.Cid
left join (select Cid,sum(case when score>0 and score<=60 then 1 else null end)[60-0],
convert(decimal(5,2),(sum(case when score>0 and score<=60 then 1 else null end))*1.00/COUNT(*))*100 所占百分比 from SC group by Cid)F on A.Cid=F.Cid;
--17. 统计各科成绩各分数段人数：课程编号，课程名称，[100-85]，[85-70]，[70-60]，[60-0] 及所占百分比 
 
select * from(select *,rank()over (partition by Cid order by score desc)A from SC)B where B.A<=3;
--18. 查询各科成绩前三名的记录（方法 1）
 
select a.Sid,a.Cid,a.score from SC a 
left join SC b on a.Cid=b.Cid and a.score<b.score
group by a.Sid,a.Cid,a.score
having COUNT(b.Sid)<3
order by a.Cid,a.score desc;
--18. 查询各科成绩前三名的记录（取 a 的最高分与本表比较）（方法 2）
 
select * from SC a where (select COUNT(*)from SC where Cid=a.Cid and score>a.score)<3
order by a.Cid,a.score desc;
--18. 查询各科成绩前三名的记录（取 a）(方法 3)
 
select Cid,COUNT(Sid)学生数 from SC group by Cid;
--19. 查询每门课程被选修的学生数 
 
select Sid,Sname from Student 
where Sid in(select Sid from(select Sid,COUNT(Cid)课程数 from SC group by Sid)A where A.课程数=2);
--20. 查询出只选修两门课程的学生学号和姓名 
 
select Ssex,COUNT(Ssex)人数 from Student group by Ssex;
--21. 查询男生、女生人数
 
select * from Student where Sname like '%风%' ;
--22. 查询名字中含有「风」字的学生信息
 
select A.*,B.同名人数 from Student A
left join (select Sname,Ssex,COUNT(*)同名人数 from Student group by Sname,Ssex)B 
on A.Sname=B.Sname and A.Ssex=B.Ssex
where B.同名人数>1;
--23. 查询同名同性学生名单，并统计同名人数
 
select * from Student where YEAR(Sage)=1990;
--24.查询 1990 年出生的学生名单
 
select Cid,AVG(score)平均成绩 from SC group by Cid order by 平均成绩 desc,Cid;
--25. 查询每门课程的平均成绩，结果按平均成绩降序排列，平均成绩相同时，按课程编号升序排列
 
select A.Sid,A.Sname,B.平均成绩 from Student A
left join (select Sid,AVG(score)平均成绩 from SC group by Sid)B on A.Sid=B.Sid
where B.平均成绩>85;
--26. 查询平均成绩大于等于 85 的所有学生的学号、姓名和平均成绩 
 
select B.Sname,A.score from(select * from SC where score<60 and Cid=(select Cid from Course where Cname='数学'))A
left join Student B on A.Sid=B.Sid;
-- 27. 查询课程名称为「数学」，且分数低于 60 的学生姓名和分数 
 
select A.Sid,B.Cid,B.score from Student A left join SC B on A.Sid=B.Sid;
-- 28. 查询所有学生的课程及分数情况（存在学生没成绩，没选课的情况）
 
select A.Sname,D.Cname,D.score from 
(select B.*,C.Cname from(select * from SC where score>70)B left join Course C on B.Cid=C.Cid)D 
left join Student A on D.Sid=A.Sid;
-- 29. 查询任何一门课程成绩在 70 分以上的姓名、课程名称和分数
 
select * from SC where score<60;
-- 30. 查询不及格的课程
 
select A.Sid,B.Sname from (select * from SC where score>80 and Cid=01)A
left join Student B on A.Sid=B.Sid;
--31. 查询课程编号为01且课程成绩在80分以上的学生的学号和姓名
 
select Cid,COUNT(*)学生人数 from SC group by Cid;
--32. 求每门课程的学生人数 
 
select top 1* from SC 
where Cid=(select Cid from Course where Tid=(select Tid from Teacher where Tname='张三')) 
order by score desc;
--33. 成绩不重复，查询选修「张三」老师所授课程的学生中，成绩最高的学生信息及其成绩
 
select *from(select *,DENSE_RANK()over (order by score desc)A 
from SC 
where Cid=(select Cid from Course where Tid=(select Tid from Teacher where Tname='张三')))B
where B.A=1;
--34. 成绩有重复的情况下，查询选修「张三」老师所授课程的学生中，成绩最高的学生信息及其成绩
 
select C.Sid,max(C.Cid)Cid,max(C.score)score from SC C 
left join(select Sid,avg(score)A from SC group by Sid)B 
on C.Sid=B.Sid
where C.score=B.A
group by C.Sid
having COUNT(0)=(select COUNT(0)from SC where Sid=C.Sid);
--35. 查询不同课程成绩相同的学生的学生编号、课程编号、学生成绩 
 
select * from
(select *,ROW_NUMBER()over(partition by Cid order by score desc)A from SC)B
where B.A<3;
--36. 查询每门功成绩最好的前两名
 
select Cid,COUNT(Sid)选修人数 from SC 
group by Cid
having COUNT(Sid)>5
order by 选修人数 desc,Cid;
--37.统计每门课程的学生选修人数（超过5人的课程才统计）。
	--要求输出课程号和选修人数，查询结果按人数降序排列，若人数相同，按课程号升序排列
 
select Sid from SC
group by Sid
having COUNT(Cid)>=2;
--38. 检索至少选修两门课程的学生学号 
 
select Sid from SC 
group by Sid 
having count(Cid)=(select distinct COUNT(0)a from Course);
--39. 查询选修了全部课程的学生信息 
 
select Sid,datediff(yy,Sage,GETDATE())年龄 from Student;
--40. 查询各学生的年龄，只按年份来算
 
select *,(case when convert(int,'1'+substring(CONVERT(varchar(10),Sage,112),5,8))
<convert(int,'1'+substring(CONVERT(varchar(10),GETDATE(),112/*112是将格式转化为yymmdd*/),5,8)) 
then datediff(yy,Sage,GETDATE()) 
else datediff(yy,Sage,GETDATE())-1 
end)age 
from Student ;
--41. 按照出生日期来算，当前月日 < 出生年月的月日则，年龄减一
	--方法是把时间转化成 Int 格式来做条件比较大小，判断是否超期，
 
select *,(case when datename(wk,convert(datetime,(convert(varchar(10),year(GETDATE()))+substring(convert(varchar(10),Sage,112),5,8))))=DATENAME(WK,GETDATE()) 
then 1 else 0 end)生日提醒
from Student;
--42. 查询本周过生日的学生
	--方法：采取将生日转化为当年日期，再转化为本年中的第几个星期进行判断搜出结果
 
select *,(case when datename(wk,convert(datetime,(convert(varchar(10),year(GETDATE()))+
substring(convert(varchar(10),Sage,112),5,8))))=DATENAME(WK,GETDATE())+1 
then 1 else 0 end)生日提醒
from Student;
--43. 查询下周过生日的学生
 
select *,(case when month(convert(datetime,(convert(varchar(10),year(GETDATE()))+substring(convert(varchar(10),Sage,112),5,8))))=month(GETDATE())
then 1 else 0 end)生日提醒
from Student;
--44. 查询本月过生日的学生
 
select *,(case when month(convert(datetime,(convert(varchar(10),year(GETDATE()))+substring(convert(varchar(10),Sage,112),5,8))))=month(GETDATE())+1
then 1 else 0 end)生日提醒
from Student;
--45. 查询下月过生日的学生