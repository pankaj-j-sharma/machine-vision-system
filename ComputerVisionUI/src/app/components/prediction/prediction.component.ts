import { Component, HostListener, OnInit } from '@angular/core';
import { AppChart } from 'src/app/interface/appchart';
import { uploadFile } from 'src/app/interface/uploadfile';
import { RestApiService } from 'src/app/services/restAPI/rest-api.service';
import LinearGradient from 'zrender/lib/graphic/LinearGradient';
import { color, EChartsOption } from 'echarts';

declare var $:any;

export interface HistoricalRun {
  RunId: string
  RunDetails : string
  ModelUsed : string
  DataSource : string
  UploadedData : string
  Prediction : string
}



@Component({
  selector: 'app-prediction',
  templateUrl: './prediction.component.html',
  styleUrls: ['./prediction.component.scss']
})
export class PredictionComponent implements OnInit {

  title = "Prediction page";
  historicalRuns : HistoricalRun[] = [];
  availableModels : any[] = [];
  availableUploadOptions : any[] = [];
  rescalingOption = "None Selected";
  selectedUseCase: string="";
  selectedModel : string = "";

  error: string= "";
  dragAreaClass: string = "";
  loadedImgUrl: string = "";
  fileList : uploadFile[] = [];
  formData: FormData = new FormData();
  appChart!: AppChart;
  backgroundColor1 : string[] = ["#00509d","#00d2ff"];
  backgroundColor2 : string[] = ["#f76775","#f993a2"];
  backgroundColor3 : string[] = ["rgb(54, 162, 235)","#80b6f4"];
  validated:boolean = false;
  processing:boolean = false;
  resourceId: number = -1;
  predictionId : number = -1;
  videoUploaded:boolean = false;
  eChartOptions!:EChartsOption ;
  arrayOfBlob = new Array<Blob>();
  emptyFile: File = new File(this.arrayOfBlob, "Mock.zip", { type: 'application/zip' });

  constructor(private restAPIService : RestApiService) { }

  // file upload code 

  @HostListener("dragover", ["$event"]) onDragOver(event: any) {
    this.dragAreaClass = "droparea";
    event.preventDefault();
  }
  @HostListener("dragenter", ["$event"]) onDragEnter(event: any) {
    this.dragAreaClass = "droparea";
    event.preventDefault();
  }
  @HostListener("dragend", ["$event"]) onDragEnd(event: any) {
    this.dragAreaClass = "dragarea";
    event.preventDefault();
  }
  @HostListener("dragleave", ["$event"]) onDragLeave(event: any) {
    this.dragAreaClass = "dragarea";
    event.preventDefault();
  }
  @HostListener("drop", ["$event"]) onDrop(event: any) {
    this.dragAreaClass = "dragarea";
    event.preventDefault();
    event.stopPropagation();
    if (event.dataTransfer.files) {
      let files: FileList = event.dataTransfer.files;
      console.log('files updated',files.length);
      this.saveFiles(files);
      this.saveFilesToPredictionData();
    }
  }

  ngOnInit(): void {
    this.loadMetaData();
  }

  loadMetaData(){
    this.processing=true;
    this.restAPIService.getAllAvailableModels({'UserId':this.restAPIService.getCurrentUserId(),'AppName':this.restAPIService.getCurrentSelectedApp()}).subscribe(resp=>{
        if ('success' in resp){
          let results = resp.results;
          for (let i = 0; i < results.length; i++) {
            this.availableModels.push(results[i]);          
          }
        }

    this.loadPredictionHistData();        
    });

    this.availableUploadOptions = [{"id":"1","name":"My Computer"},
                                    {"id":"2","name":"Cloud Drives"},
                                    {"id":"3","name":"DB Connectors"}];

                                    
    // this.historicalRuns = [
    //   {
    //     RunId:'341',
    //     RunDetails: '25_08_2021_08_55',
    //     DataSource : 'File Upload',
    //     ModelUsed : this.availableModels[0].name,
    //     UploadedData: '10 images',
    //     Prediction : '3 defected'
    //   },
    //   {
    //     RunId:'971',
    //     RunDetails: '21_08_2021_08_55',
    //     DataSource : 'File Upload',
    //     ModelUsed : this.availableModels[1].name,
    //     UploadedData: '101 images',
    //     Prediction : '13 defected'
    //   }

    // ];

    this.dragAreaClass = "dragarea";

    this.appChart=
      {
        Id:'predChart',
        Name: 'Results',
        Data:[90,10],
        Labels:["Precision","Error"],
        BgColor:this.backgroundColor1,
        Type:'pie'
      };    


  }

  loadPredictionHistData(){
    this.restAPIService.loadPredictionHistory({'UserId':this.restAPIService.getCurrentUserId() , 'AppName':this.restAPIService.getCurrentSelectedApp()}).subscribe(resp => {
      console.log('resp result',resp);
      if ('success' in resp){
        let results = resp.results;
        this.historicalRuns = [];
        for (let i = 0; i < results.length; i++) {
          this.historicalRuns.push({
            RunId:results[i].id,
            RunDetails: results[i].rundate,
            DataSource : 'File Upload',
            ModelUsed : results[i].name,
            UploadedData: results[i].FilesCount,
            Prediction : results[i].results,        
          });
        }
      }  
      this.processing=false;
    });
  }

  selectAvlModel(event:any){
    event.preventDefault();
    this.selectedUseCase = event.target.value;
    this.selectedModel = event.target.value;
    this.removeUpload();

  }

  saveFiles(files: FileList) {
    
    this.processing=true;
    this.fileList = [];
    for(let i=0; i < files.length; i++){

      this.fileList.push({'no':i,'file':files[i],'name':files[i].name,'path':'','updated':files[i].lastModified ,result:''});        
      
      // for vide files load video div
      if(files[i].name.split('.')[1]=='mp4'){
        this.videoUploaded = true;
      }

      var reader = new FileReader();

      // Closure to capture the file information.
      reader.onload = ((theFile) => {
          return (e:any) => {
            this.fileList.filter(f=>f.name==theFile.name)[0].path=e.target.result;              
          };
      })(files[i]);

      // Read in the image file as a data URL.
      reader.readAsDataURL(files[i]);        
    }

}

saveFilesToPredictionData(){
  this.formData = new FormData();
  this.formData.append('predictionId',this.predictionId.toString());
  this.formData.append('modelId',this.selectedModel.toString());
  let userId = this.restAPIService.getCurrentUserId();
  if (userId){
    this.formData.append('userId',userId.toString());
  }    
  this.formData.append('resourceId',this.resourceId.toString());
  this.fileList.forEach(f=> {
    this.formData.append(f.name,f.file);
  });

  this.restAPIService.saveFilesToPredictionData(this.formData).subscribe(resp => {
    let results = resp.results;
    if('success' in resp){
      let resultFiles = results.files;
      this.predictionId = results.predId;
      this.resourceId = results.resId;
      this.fileList = [];
      console.log('resultFiles',resultFiles,resultFiles.length);
      for(let i=0;i<resultFiles.length;i++){
        this.fileList.push({'no':i,'file':this.emptyFile,'name':resultFiles[i].name,'path':this.restAPIService.getAllSavedFiles(resultFiles[i].path),'updated':resultFiles[i].updated ,'result':resultFiles[i].result});                
      }    
    }
    this.processing= false;
  });
}

onFileChange(event:any){
  console.log('event fired ->',event)
  this.saveFiles(event.target.files);
  this.saveFilesToPredictionData();
}


onselectionChange(event:Event,change: any){
  event.preventDefault();
  this.loadMetaData(); // for dummy sample   
  console.log('e',change,event);
  let tmp = this.fileList.filter(f=>f.no==change);
  let reader = new FileReader();
  reader.onload = (e: any) => {
    this.loadedImgUrl=e.target.result;
  }
  if (tmp && tmp.length>0){
    reader.readAsDataURL(tmp[0].file);  
  }

}
removeUpload(){
  this.fileList=[];
  this.validated = false;
  this.videoUploaded = false;
  this.resourceId=-1;
  this.predictionId=-1;
  this.appChart.Type="pie";
}  

validate(event:any){
  event.preventDefault();
  this.formData = new FormData();
  this.formData.append('predictionId',this.predictionId.toString());
  this.formData.append('modelId',this.selectedModel.toString());
  this.formData.append('resourceId',this.resourceId.toString());

  let selectedModelName= this.availableModels.find(f=>f.id==this.selectedUseCase) ? this.availableModels.find(f=>f.id==this.selectedUseCase).name:'None Selected';

  this.processing=true;
  this.restAPIService.runPrediction(selectedModelName,this.formData).subscribe(result=>{
    this.validated=true;
    //this.processing=false;
    this.loadPredictionHistData();
    
    console.log('result',result);
    this.fileList.forEach(f=> {

      if(result.modelSource){
        f.path = this.restAPIService.getAllSavedFiles(result.results[f.no].FilePath);
        f.result = result.results[f.no].Tag;
      }
      // case for all pre stored models 
      else{
        f.result = result.predictions[f.no];
        if(typeof(f.result)=='object'){
          f.path = this.restAPIService.getSavedFiles(result.predictions[f.no].img);
          f.result = result.predictions[f.no].pred[0].name=='helmet' ? 'Wearing Helmet':'Without Helmet';  
        }  
      }
    });  

    // $("html, body").animate({ scrollTop: $(document).height()});

    let redChartData:any[] = [];

    this.fileList.map(f=> f.result).forEach(k=>{
      if(k.endsWith(')') && k.indexOf('(')>-1 && result.modelSource ){   
        // draw bar chart
        this.appChart.Type='bar';     
        let suff = parseInt(k.substring(k.indexOf('(')+1,k.indexOf(')')));
        for (let i =0;i < suff; i++ ){
          redChartData.push(k.substring(0,k.indexOf('(')-1));
        }
      }else{
        redChartData.push(k);
      }
    });

    let chartData = this._groupby(redChartData);
    // let chartData = this._groupby(this.fileList.map(f=> f.result));

    this.appChart.Labels=Object.keys(chartData);
    this.appChart.Data=Object.values(chartData);    

    if(this.appChart.Type=='bar'){
      this.renderEchart(this.appChart);
    }
    // this.refreshChart();
    $("html, body").animate({ scrollTop: $(document).height()});    
  });

}


__arrayMax(arr:any) {
  return arr.reduce(function (p:any, v:any) {
    return ( p > v ? p : v );
  });
}

renderEchart(appChart:AppChart){

  const dataAxis = appChart.Labels;
  const data = appChart.Data;
  const yMax = Math.floor(this.__arrayMax(appChart.Data)*(1.3));
  const dataShadow = [];
  
  for (let i = 0; i < data.length; i++) {
    dataShadow.push(yMax);
  }
  
  this.eChartOptions = {    
    title: {
      text: "",
      textStyle: {
        fontSize: "15",
      },
      left: "center",
    },
    legend: {
      show:false,
      orient: "horizontal",
      left: "right",
    },      
    tooltip: {
      trigger: "axis",
      axisPointer: {
        type: "shadow",
        animationEasingUpdate: "bounceInOut",
      },
    },
    xAxis: {      
      name: "Defect Categories",
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
    yAxis: [
      {          
        name: "Number of defects",
        nameRotate: 90,
        nameLocation: "middle",
        nameGap: 40,
        min: 0,
        maxInterval: 5,
        offset: 20,
        // max: yMax,
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
        barWidth: 80,
        barCategoryGap: '10%',
        type: "bar",
        // name: "Station 1",
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
        data:data,
      },        
    ],
  }; 

  // this.allEcharts.push(this.eChartOptions);   

}

_groupby(arr:any){
  return arr.reduce((a:any, b:any) => { 
    a[b] = a[b] || 0;
    return ++a[b], a;
  }, {});
}

removeHistory(predictionId:string){
  console.log('remove for id',predictionId);
  this.processing = true;
  this.restAPIService.removePredictionHistory({'UserId':this.restAPIService.getCurrentUserId() , 'predictionId':predictionId}).subscribe(resp => {
    if ('success' in resp){
      this.loadPredictionHistData();
    }else{
      this.processing=false;
    }
  });
  

}

fullLoadImage(imgPath:any){
  console.log('load for',imgPath)
}

}
