import { Component, OnInit } from '@angular/core';
import { Router } from '@angular/router';
import {Chart} from 'chart.js';
import { RestApiService } from 'src/app/services/restAPI/rest-api.service';

declare var jQuery:any;

export interface MyCharts{
  Id : string
  Name: string
  Data: number[]
  Labels: string[]
  BgColor:string[]
}

export interface uploadFile {
  no: number
  file : File 
  name:string
  path:string
  updated: number
};


@Component({
  selector: 'app-training',
  templateUrl: './training.component.html',
  styleUrls: ['./training.component.scss']
})
export class TrainingComponent implements OnInit {

  activationFns: any[]=[];
  customMsgs: any[]=[];  
  regulariznOptions: any[]=[];
  lossParameterOptns: any[]=[];
  optimizerOptns:any[]=[];
  categoryItems: any[] = [];
  newCategory:string='';
  testLoss:string='70';
  trainLoss:string='75';
  errored:boolean=false;
  gradientBarChartConfiguration: any;  
  canvas : any;
  ctx: any;
  myChart : any;
  backgroundColor1 : string[] = ["#00509d","#00d2ff"];
  backgroundColor2 : string[] = ["#00bcd4","#009688"];
  backgroundColor3 : string[] = ["#02216c","#00509d"];
  myDispCharts : MyCharts[]=[];
  trainingLogs:string="";
  processing:boolean=false;
  fileList : uploadFile[] = [];
  formData = new FormData();
  customTrainId:Number = -1;
  resourceId:Number = -1;
  trainType:Number = 1;
  trainingType:string="";
  val_loss:any[]=[];
  lbl_val_loss: string[]=[];
  modelFileLabel:string="my custom model 1";
  renderFormOptions:boolean=false;
  trainSuccessMessage:string = "";
  modelFilePath:string="";

  constructor(private router: Router,private restAPIService : RestApiService) { }

  ngOnInit(): void {
    // this.loadSuccessModal();
    this.loadMetaData();
    // this.myDispCharts.forEach(s=>this.generateCharts(s));
  }

  filesAdded(addedFileList:any){
    this.processing = true;
    this.fileList = addedFileList;
    console.log('added files',addedFileList);

    this.formData = new FormData();
    let userId = this.restAPIService.getCurrentUserId();
    if (userId){
      this.formData.append('userId',userId.toString());
    }

    this.formData.append('name','Custom training');
    this.formData.append('desc','Custom training details');
    this.formData.append('type',this.trainType.toString());
    this.formData.append('trainingType',this.trainingType);
    this.formData.append('resourceId',this.resourceId.toString());
    this.formData.append('trainId',this.customTrainId.toString());
    this.formData.append('customTraining','True');
    this.fileList.forEach(f=> {
      this.formData.append(f.name,f.file);
    });

    this.restAPIService.saveFilesDataForCustomTrain(this.formData).subscribe(resp => {
      if ('success' in resp){
        let results = resp.results;
        console.log('results ',results);
        this.resourceId = results['resourceId'];
        this.customTrainId = results['trainId'];
        this.appendLogs('Train images uploaded '+results['numberOfFiles']+'\n')
      }
      this.processing= false;
    });
  }  

  startTraining(data:any){
    console.log('start training ',data);
    this.processing=true;

    this.formData = new FormData();
    this.formData.append('trainId',this.customTrainId.toString());
    this.formData.append('modelFileName',this.modelFileLabel);
    this.formData.append('trainingType',data['problemtype']);
    this.formData.append('trainEpochs',data['epoch']);
    this.formData.append('learningrate',data['learningrate']);
    this.formData.append('activationAll',data['activation']);
    this.formData.append('lossparam',data['lossparam']);
    this.formData.append('optimizer',data['optimizer']);

    this.restAPIService.runCustomTraining(this.formData).subscribe(resp => {
      if ('success' in resp){
        let results = resp.results[0];
        if (results){
          if(this.trainingType=='Classification'){
            this.modelFilePath = results['model_file_path'];
            this.appendLogs('model saved successfully '+results['model_file_path']+"\n");

            // load chart data 
            this.val_loss = results['history']['val_loss'];
            this.val_loss.forEach(v=>{
              this.lbl_val_loss.push(this.val_loss.indexOf(v).toString());
            });
            // refresh chart data 
            this.myChart.data.labels = this.lbl_val_loss; 
            this.myChart.data.datasets[0].data = this.val_loss;
            this.myChart.update(); 
            this.loadSuccessModal(); 
          }

        }
        this.processing = false;
      }
    });  
  }

  removeCategory(id:any){
    this.categoryItems = this.categoryItems.filter(f=>f.id!=id);
  }

  addCategory(event:any){
    let no= Math.max.apply(Math, this.categoryItems.map(function(r) { return r.id; }));
    if(this.categoryItems.filter(f=>f.name==this.newCategory).length==0)
      this.categoryItems.push({id:no+1,name:this.newCategory});
    this.newCategory='';
  }


  initChartConfig(){

    this.gradientBarChartConfiguration = {
      maintainAspectRatio: true,
      scales: {
        xAxes: [{
            display: false,
            gridLines: {
                drawOnChartArea: false
            }
        }],
        yAxes: [{
          display: false,
          gridLines: {
                drawOnChartArea: false
            }
        }]
    },
      legend: {
        display: false,
          position: "bottom",
          labels: {
            fontColor: "#333",
            fontSize: 9
          }
      },     
        responsive: false,
        title: {
          display: false,
          position: "top",
          text: "Line Chart",
          fontSize: 13,
          fontColor: "#111"
        } 
    };

  }

  generateCharts(mychart:MyCharts){
    this.initChartConfig();

    this.canvas = document.getElementById(mychart.Id);
    this.ctx  = this.canvas.getContext("2d");
    var gradientStroke = this.ctx.createLinearGradient(0, 230, 0, 50);

    gradientStroke.addColorStop(1, 'rgba(37,138,128,0.4)');
    gradientStroke.addColorStop(0.4, 'rgba(37,138,128,0.2)');
    gradientStroke.addColorStop(0, 'rgba(37,138,128,0.1)'); //blue colors

    this.myChart = new Chart(this.ctx, {
      type: 'line',
      data: {
        labels: mychart.Labels,
        datasets: [{
          label: "Training Loss",
          fill: false,
          data: mychart.Data,          
        }]
      },
      options: this.gradientBarChartConfiguration     
    });    

  }    

  changeTrainType(event:any){
    console.log('event ->',event.target.value);
    this.clearLogs();
    if(this.trainingType=="Classification"){
      this.trainType=1;
    }
    if(this.trainingType=="Object Detection"){
      this.trainType=2;
    }

    this.formData = new FormData();
    let userId = this.restAPIService.getCurrentUserId();
    if (userId){
      this.formData.append('userId',userId.toString());
    }    
    this.formData.append('type',this.trainType.toString());
    this.restAPIService.loadDraftCustomTrainData(this.formData).subscribe(resp => {
      if ('success' in resp){
        let results = resp.results;
        console.log('results ',results);
        this.customTrainId = results['trainId'];
        this.resourceId = results['resourceId'];
        if(this.customTrainId >-1)
          this.appendLogs('custom training draft loaded successfully with id '+this.customTrainId+'\n');
      }      
    });
    this.renderFormOptions = this.trainingType!="" && this.modelFileLabel!="";
  }

  loadSuccessModal(){
    this.trainSuccessMessage = "Training completed successfully for model <b>"+this.modelFileLabel+"</b><br><br>";
    this.trainSuccessMessage+="Click here for <b> <font color ='#31c2b3'> MODEL INFERENCING</font></b>";    
    jQuery("#roleModal").modal("show");
  }

  navigateToPrediction(){
    console.log('navigate to prediction screen');
    jQuery("#roleModal").modal("hide");
    this.router.navigate([window.location.pathname.replace('training','prediction')]);
  }

  loadMetaData(){

    this.customMsgs=[
      { id:1,
        class:'',
        value:'Selected problem Type'
      },
      { id:2,
        class:'',
        value:'Upload all the image files'
      },
      { id:3,
        class:'',
        value:'Upload all the annotation files'
      },
      { id:4,
        class:'',
        value:'Upload the yml file'
      },
    ];
    this.activationFns=[
      {
        option:"Tanh",
        value:"tanh"
      },
      {
        option:"Binary Step",
        value:"binarystep"
      },
      {
        option:"ReLu",
        value:"relu"
      },
      {
        option:"Leaky ReLu",
        value:"LeakyReLU"
      },
      {
        option:"Sigmoid",
        value:"sigmoid"
      },      
    ];

    this.regulariznOptions=[
      {
        option:"L1",
        value:"1"
      },
      {
        option:"L2",
        value:"2"
      },
    ];

    this.optimizerOptns=[
      {
        option:"Adam Optimizer",
        value:"Adam"
      },
      {
        option:"RMS Propogation",
        value:"RMSprop"
      },
      {
        option:"Adaptive Gradient",
        value:"Adagrad"
      }

    ];
    this.lossParameterOptns=[
      {
        option:"Categorical Crossentropy",
        value:"categorical_crossentropy"
      },
      {
        option:"Sparse Categorical",
        value:"sparse_categorical_crossentropy"
      },
    ];

    this.categoryItems=[
      {
        id:1,
        name:'Defected'
      },
      {
        id:2,
        name:'Not Defected'
      },
    ];

    this.myDispCharts=[
        {
          Id:'trainResultChart',
          Name: 'Train Results',
          Data:[],           
          Labels:[],
          BgColor:this.backgroundColor1          
        }
    ];

    this.trainingLogs = "         --- start training your model to view the logs ---        \n"
    //this.appendLogs();

  }

  clearLogs(){
    this.trainingLogs="";  
  }  
  appendLogs(data:string){
    this.trainingLogs+=data;  
  }

  

}
