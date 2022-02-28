import { Component, OnInit } from '@angular/core';
import { FormControl } from '@angular/forms';
import { RestApiService } from 'src/app/services/restAPI/rest-api.service';
import * as Chart from 'chart.js';
import 'chartjs-plugin-labels';
import LinearGradient from 'zrender/lib/graphic/LinearGradient';
import { color, EChartsOption } from 'echarts';
import { SSL_OP_NO_TLSv1_1 } from 'constants';



@Component({
  selector: 'app-process-tracking',
  templateUrl: './process-tracking.component.html',
  styleUrls: ['./process-tracking.component.scss']
})
export class ProcessTrackingComponent implements OnInit {

  searchClicked:boolean=false;
  searchResults:any[] = [];
  leftSidePnlOptions : any[] =[];
  playlist : any[]=[];
  baseVideoAssetsPath= "../../../assets/videos/";
  visiblePlaylist : any[] = [];
  selectedPlaylist : any;
  panelOpenState = false;
  myControl = new FormControl();
  currentIndex = 0;
  currentItem :any;
  activeTab:string="Stored Videos";
  selectedPartNo:string="";
  items: number[]=[];
  ckselected:number=-1;
  formdata:FormData = new FormData();

  // filter options
  recentSearches : any[]=[];
  dataSources : any[]=[];
  activeStations : any[]=[];
  productNos : any[]=[];
  partNos: any[]= [];
  avlShifts:any[]=[];
  kpiMetrics:any[]=[];
  timeRangeVal :number = 0;
  navigatedFrom:string="";

  // echarts 
  allEcharts:any[]=[];
  eChartOptions!:EChartsOption ;
  // echarts

  // chart option starts 
  gradientBarChartConfiguration: any;  
  canvas : any;
  ctx: any;
  myChart : any;
  bgColorArr = ["#00509d", "#00d2ff","#f76775", "#f993a2","rgb(54, 162, 235)", "#80b6f4"];

  constructor(private restAPIService : RestApiService) { }

  ngOnInit(): void {
    this.loadVideoData();
    this.renderEChart();
  }

  renderEChart(){
    this.renderEChart1();
    this.renderEChart2();
    this.renderEChart3();    
    this.renderEChart4();
  }


  renderEChart1(){
    let color = ["#ffffff", "#02b8ac", "#374649"];

    const dataAxis = [
      "06-Jan",
      "07-Jan",
      "08-Jan",
      "09-Jan",
      "10-Jan",
      "11-Jan",
      "12-Jan",
      "13-Jan",
      "14-Jan",
    ];
    const data = [612, 630, 615, 650, 621, 637, 620, 679, 663];
    const yMax = 800;
    const dataShadow:any[] = [];
    
    for (let i = 0; i < data.length; i++) {
      //dataShadow.push(yMax);
    }
    
    this.eChartOptions = {
      // grid: {
      //   width: '50%',
      //   height:'50%',
      //   left: '3%',
      //   right: '4%',
      //   bottom: '3%',
      //   containLabel: true
      // },
      title: {
        text: "Cycles Count and Mean Variations per day",
        textStyle: {
          fontSize: "15",
        },
        left: "center",
      },
      legend: {
        data: ["Cycles completed", "Mean Cycle Time"],
        orient: "horizontal",
        left: "right",
      },
      tooltip: {
        trigger: "axis",
        axisPointer: {
          type: "shadow",
          animationEasingUpdate: "bounceInOut",
        },
        formatter: function (params: any) {
          let colorSpan = (color: string) =>
            '<span style="display:inline-block;margin-right:5px;border-radius:10px;width:9px;height:9px;background-color:' +
            color +
            '"></span>';
          let rez = "<p style='margin-bottom:0px'>" + params[0].axisValue + "</p>";
          params.forEach((item: any) => {
            // console.log('item',item); //quite useful for debug
            if (item.seriesName != "shadow") {
              var xx =
                "<p style='margin-bottom:0px'>" +
                colorSpan(color[item.seriesIndex]) +
                " " +
                item.seriesName +
                ": " +
                item.data +
                "</p>";
              rez += xx;
            }
          });
          return rez;
        },
      },
      // toolbox: {
      //   feature: {
      //     dataView: { show: true, readOnly: false },
      //     restore: { show: true },
      //     saveAsImage: { show: true }
      //   }
      // },
      xAxis: {
        name: "Working days",
        nameLocation: "middle",
        nameGap: 30,
        data: dataAxis,
        axisLabel: {
          align: "center",
          color: "#999",
        },
        axisTick: {
          show: false,
        },
        axisLine: {
          show: false,
        },
        z: 10,
      },
      yAxis: [
        {
          name: "Total cycles completed",
          nameRotate: 90,
          nameLocation: "middle",
          nameGap: 40,
          min: 0,
          maxInterval: 100,
          offset: 20,
          max: yMax,
          splitLine: {
            show: false,
          },
          axisLine: {
            show: false,
          },
          axisTick: {
            show: false,
          },
          axisLabel: {
            color: "#999",
          },
        },
        {
          name: "Mean Cycle Time ( in sec )",
          nameRotate: 90,
          nameLocation: "middle",
          nameGap: 40,
          min: 55,
          max: 90,
          splitLine: {
            show: false,
          },
          offset: 20,
          position: "right",
        },
      ],
      // dataZoom: [
      //   {
      //     type: 'inside',
      //   },
      // ],
      series: [
        {
          // For shadow
          type: "bar",
          name: "shadow",
          itemStyle: {
            color: "rgba(0,0,0,0.05)",
          },
          barGap: "-100%",
          barCategoryGap: "40%",
          data: dataShadow,
          animation: false,
        },
        {
          type: "bar",
          name: "Cycles completed",
          label: {
            show: false,
            align: "center",
            distance: 15,
            position: "insideBottom",
            color: "#fff",
          },
          itemStyle: {
            color: new LinearGradient(0, 0, 0, 1, [
              { offset: 0, color: color[1] },
              { offset: 0.5, color: color[1] },
              { offset: 1, color: color[1] },
            ]),
          },
          emphasis: {
            itemStyle: {
              color: new LinearGradient(0, 0, 0, 1, [
                { offset: 0, color: color[1] },
                { offset: 0.7, color: color[1] },
                { offset: 1, color: color[1] },
              ]),
            },
          },
          data,
        },
        {
          name: "Mean Cycle Time",
          type: "line",
          lineStyle:{
            width:2,
            type: 'dashed'
          },
          label: {
            show: false,
            align: "center",
            distance: 15,
            position: "insideBottom",
            color: "#fff",
          },
          itemStyle: {
            color: new LinearGradient(0, 0, 0, 1, [
              { offset: 0, color: color[2] },
              { offset: 0.5, color: color[2] },
              { offset: 1, color: color[2] },
            ]),
          },
          emphasis: {
            itemStyle: {
              color: new LinearGradient(0, 0, 0, 1, [
                { offset: 0, color: color[2] },
                { offset: 0.7, color: color[2] },
                { offset: 1, color: color[2] },
              ]),
            },
          },
          data: [65, 70, 68, 64, 61, 79, 73, 70, 62],
          yAxisIndex: 1,
        },
      ],
    }; 
    this.allEcharts.push(this.eChartOptions);   
  }
  
  renderEChart2(){
    const dataAxis = [
      "06-Jan",
      "07-Jan",
      "08-Jan",
      "09-Jan",
      "10-Jan",
      "11-Jan",
      "12-Jan",
      "13-Jan",
      "14-Jan",
    ];
    const data = [612, 630, 615, 650, 621, 637, 620, 679, 663];
    const yMax = 700;
    const dataShadow = [];
    
    for (let i = 0; i < data.length; i++) {
      dataShadow.push(yMax);
    }
    
    this.eChartOptions = {
      title: {
        text: "Cycle Mean Variations per Station",
        textStyle: {
          fontSize: "15",
        },
        left: "center",
      },
      legend: {
        orient: "horizontal",
        left: "right",
      },

      // toolbox: {
      //   top:20 ,
      //   right:10,
      //   feature: {
      //     dataZoom: {
      //       yAxisIndex: 'none'
      //     },
      //     restore: {},
      //     saveAsImage: {}
      //   }
      // },
      
      tooltip: {
        trigger: "axis",
        axisPointer: {
          type: "shadow",
          animationEasingUpdate: "bounceInOut",
        },
        formatter: function (params: any) {
          let color = ["#02b8ac", "#384749", "#fc655d"];
          let colorSpan = (color: string) =>
            '<span style="display:inline-block;margin-right:5px;border-radius:10px;width:9px;height:9px;background-color:' +
            color +
            '"></span>';
          let rez = "<p style='margin-bottom:0px'>" + params[0].axisValue + "</p>";
          params.forEach((item: any) => {
            // console.log('item',item); //quite useful for debug
            if (item.seriesName != "shadow") {
              var xx =
                "<p style='margin-bottom:0px'>" +
                colorSpan(color[item.seriesIndex]) +
                " " +
                item.seriesName +
                ": " +
                item.data +
                "</p>";
              rez += xx;
            }
          });
          return rez;
        },
      },
      xAxis: {
        name: "Working days",
        nameLocation: "middle",
        nameGap: 57,
        data: dataAxis,
        axisLabel: {
          align: "center",
          color: "#999",
        },
        axisTick: {
          show: false,
        },
        axisLine: {
          show: false,
        },
        z: 10,
      },
      dataZoom: [{
        type: 'slider',
        start: 0,
        end: Math.round((8*100)/dataAxis.length),
        realtime:true,
        bottom:30,
        height:20
      }],    
      yAxis: [
        {          
          name: "Total cycles completed",
          nameRotate: 90,
          nameLocation: "middle",
          nameGap: 40,
          min: 0,
          maxInterval: 100,
          offset: 20,
          max: yMax,
          splitLine: {
            show: true,
          },
          axisLine: {
            show: false,
          },
          axisTick: {
            show: false,
          },
          axisLabel: {
            color: "#999",
          },
        }
      ],
      series: [
        {
          type: "bar",
          name: "Station 1",
          label: {
            show: false,
            align: "center",
            distance: 15,
            position: "insideBottom",
            color: "#fff",
          },
          itemStyle: {
            color: new LinearGradient(0, 0, 0, 1, [
              { offset: 0, color: "#02b8ac"  },
              { offset: 0.5, color: "#02b8ac" },
              { offset: 1, color: "#02b8ac" },
            ]),
          },
          emphasis: {
            itemStyle: {
              color: new LinearGradient(0, 0, 0, 1, [
                { offset: 0, color: "#02b8ac" },
                { offset: 0.7, color: "#02b8ac" },
                { offset: 1, color: "#02b8ac" },
              ]),
            },
          },
          data:[512, 550, 511, 590, 487, 451, 491, 534, 555],
        },
        {
          type: "bar",
          name: "Station 2",
          label: {
            show: false,
            align: "center",
            distance: 15,
            position: "insideBottom",
            color: "#fff",
          },
          itemStyle: {
            color: new LinearGradient(0, 0, 0, 1, [
              { offset: 0, color: "#384749" },
              { offset: 0.5, color: "#384749" },
              { offset: 1, color: "#384749" },
            ]),
          },
          emphasis: {
            itemStyle: {
              color: new LinearGradient(0, 0, 0, 1, [
                { offset: 0, color: "#384749" },
                { offset: 0.7, color: "#384749" },
                { offset: 1, color: "#384749" },
              ]),
            },
          },
          data:[412, 450, 411, 390, 387, 351, 391, 434, 355],
        },

        {
          type: "bar",
          name: "Station 3",
          label: {
            show: false,
            align: "center",
            distance: 15,
            position: "insideBottom",
            color: "#fff",
          },
          itemStyle: {
            color: new LinearGradient(0, 0, 0, 1, [
              { offset: 0, color: "#fc655d" },
              { offset: 0.5, color: "#fc655d" },
              { offset: 1, color: "#fc655d" },
            ]),
          },
          emphasis: {
            itemStyle: {
              color: new LinearGradient(0, 0, 0, 1, [
                { offset: 0, color: "#fc655d" },
                { offset: 0.7, color: "#fc655d" },
                { offset: 1, color: "#fc655d" },
              ]),
            },
          },
          data:[212, 250, 211, 190, 287, 251, 191, 234, 155],
        },
        
      ],
    }; 

    this.allEcharts.push(this.eChartOptions);   
  }


  renderEChart3(){
    let color = ['#585858', '#fd625e', '#7ae1d9', '#efb7b5'];
    // color = ["#908654","#de690c","#bfb792","#f2ba8b"]

    const dataAxis = [
      "06-Jan",
      "07-Jan",
      "08-Jan",
      "09-Jan",
      "10-Jan",
      "11-Jan",
      "12-Jan",
      "13-Jan",
      "14-Jan",
      "15-Jan",
      "16-Jan",
      "17-Jan",
      "18-Jan",
      "19-Jan",
      "20-Jan",
      "21-Jan",
      "22-Jan",
      "23-Jan",
    ];

    // random data generation
    let yMax = 100;
    let categoryData = [];
    let errorData:any[] = [];
    let barData:any[] = [];
    let barData1:any[] = [];
    let dataCount = dataAxis.length;

    let categoryData2:any[] = [];
    let errorData2:any[] = [];
    let barData2:any[] = [];
    let barData3:any[] = [];
    let dataCount2 = dataAxis.length;
    
    for (let i = 0; i < dataCount; i++) {
      let val = Math.random() * (yMax - 0.08*yMax);
      let outView = parseFloat(Math.max(3,Math.random()*val - 0.7*val).toFixed(2)) ;
      let inView = parseFloat((val - outView).toFixed(2));
      categoryData.push(dataAxis[i]);
      errorData.push([
        i,
        Math.round(Math.max(0, inView - Math.random() * (0.08*yMax))),
        Math.round(inView + Math.random() * (0.04*yMax))
      ]);
      barData.push(inView);
      barData1.push(outView);
    }    

    for (let i = 0; i < dataCount2; i++) {
      const val = Math.random() * (yMax - 0.08*yMax);;
      let outView = parseFloat(Math.max(3,Math.random()*val - 0.7*val).toFixed(2)) ;
      let inView = parseFloat((val - outView).toFixed(2));
      categoryData2.push(i);
      errorData2.push([
        i,
        Math.round(Math.max(0, inView - Math.random() * (0.08*yMax))),
        Math.round(inView + Math.random() * (0.03*yMax))
      ]);
      barData2.push(inView);
      barData3.push(outView);
    }
    
  
    // random data generation

    this.eChartOptions = {
      title: {
        text: "Unit in-view and out-of-view time per Station",
        textStyle: {
          fontSize: "15",
        },
        left: "center",
      },
      legend: {
        orient: "horizontal",
        left: "center",
        top:30,
        data:['Station 1 : in-view time','Station 1 : out-of-view time','Station 2 : in-view time','Station 2 : out-of-view time']
      },
      tooltip: {
        trigger: "axis",
        axisPointer: {
          type: "shadow",
          animationEasingUpdate: "bounceInOut",
        },
        formatter: function (params: any) {
          let colorSpan = (color: string) =>
            '<span style="display:inline-block;margin-right:5px;border-radius:10px;width:9px;height:9px;background-color:' +
            color +
            '"></span>';
          let rez = "<p style='margin-bottom:0px'>" + params[0].axisValue + "</p>";
          params.forEach((item: any) => {
            // console.log('item',item); //quite useful for debug
            if (item.seriesName != "shadow") {
              var xx =
                "<p style='margin-bottom:0px'>" +
                colorSpan(color[item.seriesIndex]) +
                " " +
                item.seriesName +
                ": " +
                item.data +
                "</p>";
              rez += xx;
            }
          });
          return rez;
        },
      },
      dataZoom: [{
        type: 'slider',
        start: 0,
        end: Math.round((8*100)/dataAxis.length),
        realtime:true,
        bottom:30,
        height:20
      }],    
      xAxis: {
        data: categoryData,
        name: "Working days",
        nameLocation: "middle",
        nameGap:57,
        axisLabel: {
          align: "center",
          color: "#999",
        },
        axisTick: {
          show: false,
        },
        axisLine: {
          show: false,
        },
        z: 10,      
      },    
      yAxis: [
        {          
          name: "Total cycles completed",
          nameRotate: 90,
          nameLocation: "middle",
          nameGap: 40,
          min: 0,
          maxInterval: 100,
          offset: 20,
          max: yMax,
          splitLine: {
            show: false,
          },
          axisLine: {
            show: false,
          },
          axisTick: {
            show: false,
          },
          axisLabel: {
            color: "#999",
          },
        }

      ],    
      series: [{
        type: 'bar',
        name: 'Station 1 : in-view time',
        stack:"Station 1",
        data: barData,
        itemStyle: {
          color: color[0]
        }
        }, {
          type: 'bar',
          name: 'Station 2 : in-view time',
          stack:"Station 2",
          data: barData2,
          itemStyle: {
            color: color[1]
          }
        },
        
        {
          type: 'bar',
          name: 'Station 1 : out-of-view time',
          stack:"Station 1",
          data: barData1,
          itemStyle: {
            color: color[2]
          }
        },
        {
          type: 'bar',
          name: 'Station 2 : out-of-view time',
          stack:"Station 2",
          data: barData3,
          itemStyle: {
            color: color[3]
          }
        },
        {
          type: 'custom',
          name: 'error - Station 1',
          itemStyle: {
            borderWidth:2
          },
          renderItem: function (params:any, api:any) {
            const errorGroup = 0;
            const numErrorGroups = 2;
            const xValue = api.value(0);
            const highPoint = api.coord([xValue, api.value(1)]);
            const lowPoint = api.coord([xValue, api.value(2)]);
            
            const categoryWidth = api.size([1, 0])[0];
            
            const barGap = categoryWidth * 0.05;
            const barWidth = (categoryWidth / numErrorGroups) - barGap;
            highPoint[0] = highPoint[0] - (barWidth / 2) + (errorGroup * barWidth);
            lowPoint[0] = highPoint[0];
            
            const errorBarWidth = categoryWidth * 0.05;
            
            const style = api.style({
              // stroke: api.visual('color'),
              stroke:'white',
              fill: null
            });
          
            return {
              type: 'group',
              children: [{
                type: 'line',
                shape: {
                  x1: highPoint[0] - errorBarWidth, y1: highPoint[1],
                  x2: highPoint[0] + errorBarWidth, y2: highPoint[1]
                },
                style: style
              }, {
                type: 'line',
                shape: {
                  x1: highPoint[0], y1: highPoint[1],
                  x2: lowPoint[0], y2: lowPoint[1]
                },
                style: style
              }, {
                type: 'line',
                shape: {
                  x1: lowPoint[0] - errorBarWidth, y1: lowPoint[1],
                  x2: lowPoint[0] + errorBarWidth, y2: lowPoint[1]
                },
                style: style
              }]
            };
          },
          data: errorData,
          z: 100
        }, {
          type: 'custom',
          name: 'error - Station 2',
          itemStyle: {
            borderWidth:2
          },
          renderItem: function (params:any, api:any) {
            const errorGroup = 1;
            const numErrorGroups = 2;
            const xValue = api.value(0);
            const highPoint = api.coord([xValue, api.value(1)]);
            const lowPoint = api.coord([xValue, api.value(2)]);
            
            const categoryWidth = api.size([1, 0])[0];
            
            const barGap = categoryWidth * 0.05;
            const barWidth = (categoryWidth / numErrorGroups) - barGap;
            highPoint[0] = highPoint[0] - (barWidth / 2) + (errorGroup * barWidth);
            lowPoint[0] = highPoint[0];
            
            const errorBarWidth = categoryWidth * 0.05;
            
            const style = api.style({
              // stroke: api.visual('color'),
              stroke:'white',
              fill: null
            });
          
            return {
              type: 'group',
              children: [{
                type: 'line',
                shape: {
                  x1: highPoint[0] - errorBarWidth, y1: highPoint[1],
                  x2: highPoint[0] + errorBarWidth, y2: highPoint[1]
                },
                style: style
              }, {
                type: 'line',
                shape: {
                  x1: highPoint[0], y1: highPoint[1],
                  x2: lowPoint[0], y2: lowPoint[1]
                },
                style: style
              }, {
                type: 'line',
                shape: {
                  x1: lowPoint[0] - errorBarWidth, y1: lowPoint[1],
                  x2: lowPoint[0] + errorBarWidth, y2: lowPoint[1]
                },
                style: style
              }]
            };
          },
          data: errorData2,
          z: 100
        }]      
      }
    this.allEcharts.push(this.eChartOptions);   
  }

  renderEChart4(){
    const dataAxis = [
      "06-Jan",
      "07-Jan",
      "08-Jan",
      "09-Jan",
      "10-Jan",
      "11-Jan",
      "12-Jan",
      "13-Jan",
      "14-Jan",
      "15-Jan",
      "16-Jan",
      "17-Jan",
      "18-Jan",
      "19-Jan",
      "20-Jan",
      "21-Jan",
      "22-Jan",
      "23-Jan",
    ];
    let yMax = 100 ;
    let myData = [];
    let color = ['#585858', '#0e2e2c'];

    for (let i = 0; i < 5; i++) {
      let tmp:any[] = [];
      for (let i = 0; i < 20; i++) {
        if(Math.random() > 0.7){
          tmp.push(Math.round(Math.random() * (yMax)));
        }else{
          tmp.push(Math.round(Math.random() * (yMax*.3)));
        }
      }
      myData.push(tmp);
    }

    this.eChartOptions = {
      title: [
        {
          text: 'Station variability time',
          textStyle: {
            fontSize: "15",
          },  
          left: 'center'
        },
        {
          text: 'upper: Q3 + 1.5 * IQR \nlower: Q1 - 1.5 * IQR',
          borderColor: '#999',
          borderWidth: 1,
          textStyle: {
            fontWeight: 'normal',
            fontSize: 14,
            lineHeight: 20
          },
          left: '10%',
          top: '90%'
        }
      ],
      dataset: [
        {
          source: myData
        },
        {
          transform: {
            type: 'boxplot',
            config: { itemNameFormatter: function(params:any){ 
              return 'Station '+parseInt(params.value+1);
            } }
          }
        },
        {
          fromDatasetIndex: 1,
          fromTransformResult: 1
        }
      ],
      tooltip: {
        trigger: 'item',
        axisPointer: {
          type: 'shadow'
        },
        formatter: function (param:any) {
          console.log('param',param);
          let seriesName = ['name','min','Q1','median','Q3','max']
          let colorSpan = (color: string) =>
            '<span style="display:inline-block;margin-right:5px;border-radius:10px;width:9px;height:9px;background-color:' +
            color +
            '"></span>';
          let rez = '<p style="margin-bottom:0px">' + param.seriesName + '</p>';
    
          if(param.seriesIndex == 0){
            seriesName.forEach((item,index)=> {
            rez +=
              '<p style="margin-bottom:0px">' + colorSpan(color[param.seriesIndex]) +
              ' '+item+'  <span style="float:right;margin-left:15px;font-weight:900">' + param.data[index] +
              '</span></p>';        
            });
          }else{
            rez +=
              '<p style="margin-bottom:0px">' + colorSpan(color[param.seriesIndex]) +
              ' '+param.data[0]+' <span style="float:right;margin-left:15px;font-weight:900">' + param.data[1] +
              '</span></p>';        
          }
          return rez;
        }
    
      },
      grid: {
        left: '10%',
        right: '10%',
        bottom: '15%'
      },
      xAxis: {
        type: 'category',
        boundaryGap: true,
        nameGap: 30,
        // splitArea: {
        //   show: false
        // },
        splitLine: {
          show: true
        }
      },
      yAxis: {
        type: 'value',
        name: 'Cycle time in seconds',
        nameRotate: 90,
        nameLocation: "middle",
        nameGap: 40,
        min: 0,
        maxInterval: 100,
        offset: 20,
        max: 100,
        splitLine: {
          show: true,
        },
        axisLine: {
          show: false,
        },
        axisTick: {
          show: false,
        },
        axisLabel: {
          color: "#999",
        },

      },
      series: [
        {
          name: 'boxplot',
          type: 'boxplot',
          itemStyle:{
            color:'#585858',
            borderColor:'#01b8aa',
            borderWidth:2
          },    
          datasetIndex: 1
        },
        {
          name: 'outlier',
          type: 'scatter',
          datasetIndex: 2,
          symbolSize:8,
          itemStyle:{
          borderColor:'#585858',
          borderWidth:0.5,
          color:'#585858',
          },
        }
      ]
    };
    this.allEcharts.push(this.eChartOptions);   
  }

  onChartEvent(event: any, type: string) {
    console.log('chart event:', type, event);
  }

  __onlyUnique(value:any, index:number, self:any) {
    return self.indexOf(value) === index;
  }

  applyFilterOptions(data:any){
    console.log('data',data)
  }

  navigateToAnalytics(selectedPart:string){
    console.log('selectedPart',selectedPart);
    this.searchClicked = true;
    this.activeTab='Video Analytics';
    this.navigatedFrom="Stored Videos";    
  }

  navigateBack(event:any){
    event.preventDefault();
    this.activeTab='Stored Videos';
    this.navigatedFrom="";
  }

  onFilterInputChange(event:any,type:string){
    if(type=='checkbox'){
      console.log('eve', event.target.checked,event.target.name);
    }
    if(type=="slider"){
      console.log('eve', event.target.value,event.target.name);
    }
  }

  loadVideoData(){
    let userId:string = this.restAPIService.getCurrentUserId() || "-1";
    this.formdata.append('userId',userId);

    this.restAPIService.loadAllProcessVideos(this.formdata).subscribe(resp =>{
      if ('success' in resp){
        let results = resp['results'];
        this.playlist = results;
        this.playlist.forEach(p => {

          this.restAPIService.getS3BucketUrl({'fileName':p.Thumbnail}).subscribe(respUrl=>{
            // console.log('passed Thumbnail ',p.Thumbnail,respUrl);
            if (respUrl){
              p.Thumbnail = respUrl.s3Url;
            }else{
              p.Thumbnail=this.restAPIService.getAllSavedFiles(p.Thumbnail);
            }
          });

          this.restAPIService.getS3BucketUrl({'fileName':p.Source}).subscribe(respUrl=>{
            // console.log('passed source ',p.Source,respUrl);
            if (respUrl){
              p.Source = respUrl.s3Url;
            }else{
              p.Source = this.restAPIService.getAllSavedFiles(p.Source);
            }
          });
        });
        this.currentItem= this.playlist[this.currentIndex];
        this.selectedPartNo = this.currentItem.PartNo;

        this.dataSources= this.playlist.map(p => p.DataSource).filter(this.__onlyUnique);
        this.activeStations= this.playlist.map(p => p.ActiveStation).filter(this.__onlyUnique);
        this.productNos= this.playlist.map(p => p.ProductNo).filter(this.__onlyUnique);
        this.partNos= this.playlist.map(p => p.PartNo).filter(this.__onlyUnique);
        this.avlShifts = ['09 AM - 05 PM','05 PM - 01 AM','01 AM - 09 AM'];
        this.kpiMetrics = ["Cycles completed","Stationwise cycle time","Unit in/out view time","Station Variability"]
      }
    })

    this.leftSidePnlOptions=[
      {
        Title:'Recent Searches',
        Body: 'Selected range'
      },{
        Title:'Time Range',
        Body: 'Selected range'        
      },
      {
        Title:'Cycle Time',
        Body: 'Selected range'
      },
      {
        Title:'Data Source',
        Body: [{
          chkbox:'Camera Feed',
          val:1
        },{
          chkbox:'Visimatic Continuos',
          val:2
        },{
          chkbox:'Self Annotated',
          val:3
        }
        ]
      },
      {
        Title:'Active Stations',
        Body: [{
          chkbox:'Station 1091.1A #34213451',
          val:1
        },{
          chkbox:'Station 1091.2A #32123670',
          val:2
        },{
          chkbox:'Station 4120.1T #72923670',
          val:3
        },{
          chkbox:'Station 4120.2T #71209876',
          val:4
        },{
          chkbox:'Station 3121.1Q #41133389',
          val:5
        }
        ]
      },
      {
        Title:'Products',
        Body: [{
          chkbox:'JSDF QWE 1WQF QSZ3',
          val:1
        },{
          chkbox:'2WDF UI8R 4XZ1 P0UA',
          val:2
        },{
          chkbox:'SQ12 5TT1 KJA1 X12W',
          val:3
        },{
          chkbox:'CVX1 23WS AS1Q PLKZ',
          val:4
        },{
          chkbox:'Z123 67YU 09GH 12QW',
          val:5
        },{
          chkbox:'ZXCV 123F YTRE ASDF',
          val:6         
        }
        ]
      },

    ];
    this.visiblePlaylist = this.playlist;

  }

  onClickPlaylistItem(item:any){
    console.log('changed to',item.Source);
    this.currentItem = item;
    this.selectedPartNo = this.currentItem.PartNo;
  }

  searchVideos(data:any){
    this.searchClicked=true;
    // this.generateCharts();
  }

  selectResultOption(id:number){
    console.log('id ..',id);    
  }

  loadResults(keyword:any){
    console.log('searched ..',keyword);
    this.searchResults=[
      {
        id:1,
        title:'Search 1',
        description:'result 1'
      },
      {
        id:2,
        title:'Search 2',
        description:'result 1'
      },
      {
        id:3,
        title:'Search 3',
        description:'result 1'
      },
    ]
    this.searchResults = this.searchResults.filter(f=> f.title.includes(keyword) || f.description.includes(keyword));

  }
}
