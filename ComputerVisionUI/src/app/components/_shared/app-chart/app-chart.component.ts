import { AfterViewInit, Component, Input, OnInit, SimpleChanges } from '@angular/core';
import * as Chart from 'chart.js';
import { AppChart } from 'src/app/interface/appchart';
import 'chartjs-plugin-labels';


@Component({
  selector: 'app-chart',
  templateUrl: './app-chart.component.html',
  styleUrls: ['./app-chart.component.scss']
})
export class AppChartComponent implements OnInit,AfterViewInit {

  gradientBarChartConfiguration: any;  
  canvas : any;
  ctx: any;
  myChart : any;
  bgColorArr = ["#00509d", "#00d2ff","#f76775", "#f993a2","rgb(54, 162, 235)", "#80b6f4"];
  
  @Input()
  appChart!: AppChart;
    
  constructor() { }
  ngAfterViewInit(): void {
    if(this.appChart.Type=='bar-e'){
      this.generateEchart(this.appChart);
    }else{
      this.generateCharts(this.appChart);
    }
  }

  ngOnInit(): void {
    this.loadAllMetaData();
  }

  ngOnChanges(changes: SimpleChanges) {  
    console.log('changes ',changes);
    // this.generateCharts(this.appChart);
  }

  loadAllMetaData(){
  }

  generateEchart(appChart:AppChart){

  }

  initChartConfig(){

    this.gradientBarChartConfiguration = {
      tooltips: {
        enabled: false
      },      
      maintainAspectRatio: true,
      legend: {
        display: true,
          position: "right",
          labels: {
            fontColor: "#333",
            fontSize: 19
          }
      },     
        responsive: true,
        title: {
          display: false,
          position: "top",
          text: "Prediction Results",
          fontSize: 21,
          fontColor: "#111"
        },
        plugins: {
          labels: {
            render: 'percentage',
            // position:'outside',
            textShadow: true,
            fontSize: 19,
            fontColor: "white",
            precision: 2,
            tooltip: {
              enabled: false
            },                      
          }
        },        
    };

  }

  generateCharts(appChart:AppChart){
    console.log('appchart',appChart);
    this.initChartConfig();    
    this.canvas = document.getElementById(appChart.Id);
    this.ctx  = this.canvas.getContext("2d");
    var gradientStroke = this.ctx.createLinearGradient(0, 230, 0, 50);

    gradientStroke.addColorStop(1, 'rgba(37,138,128,0.4)');
    gradientStroke.addColorStop(0.4, 'rgba(37,138,128,0.2)');
    gradientStroke.addColorStop(0, 'rgba(37,138,128,0.1)'); //blue colors


    this.myChart = new Chart(this.ctx, {
      type: appChart.Type,
      data: {        
        labels: appChart.Labels,
        datasets: [{
          label: "Predictions",
          fill: true,
          //backgroundColor: appChart.BgColor,
          backgroundColor : appChart.Type=='bar'? this.bgColorArr[Math.floor(Math.random()*this.bgColorArr.length)] :this.bgColorArr,
          borderColor: 'white',
          borderWidth: 1,
          borderDash: [],
          borderDashOffset: 0.0,
          data: appChart.Data,
          
        }]
      },
      options: this.gradientBarChartConfiguration     
    });    

  }  


}
