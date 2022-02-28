const iconBaseUrl:string='../../../assets/images/login/icon/';

const homePageAppItems = [
    {
      appName:'defectdetection',
      appDescription:'Defect Detection',
      iconUrl:iconBaseUrl+'Defect-Detection-w-new.png'
    },
    {
      appName:'videoanalytics',
      appDescription:'Video Analytics',
      iconUrl:iconBaseUrl+'Video-Analytics-w.png'
    },
    {
      appName:'processtracking',
      appDescription:'Process Tracking',
      iconUrl:iconBaseUrl+'end-to-end-lifecycle-w.png'
    },
    {
      appName:'intrusiondetection',
      appDescription:'Intrusion Detection',
      iconUrl:iconBaseUrl+'Intrusion-Detection-w.png'
    },
    {
      appName:'headcountprediction',
      appDescription:'Head Count Prediction',
      iconUrl:iconBaseUrl+'Head-Count-Prediction-w.png'
    },
    {
      appName:'facialrecognition',
      appDescription:'Facial Recognition',
      iconUrl:iconBaseUrl+'Facial-Recognition-w.png'
    },
    {
      appName:'safetycompliance',
      appDescription:'Safety Compliance',
      iconUrl:iconBaseUrl+'Safety-Compliance-w.png'
    },
    {
      appName:'vehicleparking',
      appDescription:'Vehicle Parking',
      iconUrl:iconBaseUrl+'Vehicle-Parking-w.png'
    },
    {
      appName:'labeldetection',
      appDescription:'Label Detection',
      iconUrl:iconBaseUrl+'Label-Detection-w.png'
    },
    {
      appName:'damagedetection',
      appDescription:'Damage Detection',
      iconUrl:iconBaseUrl+'Damage-Detection-w.png'
    },
    {
      appName:'barcodeanalysis',
      appDescription:'Barcode Analysis',
      iconUrl:iconBaseUrl+'barcode-analysis-w.png'
    },
  ]

const navSideBarItem= [
    {
      title:'Home',
      description:'Home',
      iconClass:'fa fa-fw bg-home',
      url:'/home',
      roles:['bu','ds'],
      subItems:[]
    },
    {
      title:'Project',
      description:'Project',
      iconClass:'fa fa-fw bg-project',
      url:'/project',
      roles:['ds'],
      subItems:[
        // {
        //   title:'Navbar',
        //   description:'Navbar',
        //   iconClass:'',
        //   url:'',
        //   roles:['ds'],
        //   subItems:[]    
        // },
        // {
        //   title:'Cards',
        //   description:'Cards',
        //   iconClass:'',
        //   url:'',
        //   roles:['ds'],
        //   subItems:[]    
        // }
      ]
    },
    {
      title:'Prediction',
      description:'Prediction',
      iconClass:'fa fa-fw bg-pridiction',
      url:'/prediction',
      roles:['bu','ds'],
      subItems:[]
    },
    {
      title:'Training',
      description:'Training',
      iconClass:'fa fa-fw bg-model',
      url:'/training',
      roles:['ds'],
      subItems:[]
    },
    {
      title:'Labelling',
      description:'Labelling',
      iconClass:'fa fa-fw bg-label',
      url:'/labelling',
      roles:['ds'],
      subItems:[]
    },
    {
      title:'Augmentation',
      description:'Augmentation',
      iconClass:'fa fa-fw bg-aurg',
      url:'\/augmentation',
      roles:['ds'],
      subItems:[

        // {
        //   title:'Second Level First',
        //   description:'Second Level First',
        //   iconClass:'',
        //   url:'augmentationmenu11',
        //   roles:['ds'],
        //   subItems:[
        //     {
        //       title:'Third Level 2',
        //       description:'Third Level 2',
        //       iconClass:'',
        //       url:'',
        //       roles:['ds'],
        //       subItems:[]    
        //     }    
        //   ]    
        // },
        // {
        //   title:'Second Level 2',
        //   description:'Second Level 2',
        //   iconClass:'',
        //   url:'',
        //   roles:['ds'],
        //   subItems:[]    
        // }

      ]
    },
    {
      title:'Data-Sampling',
      description:'Data Sampling',
      iconClass:'fa fa-fw bg-datasample',
      url:'\/sampling',
      roles:['ds'],      
      subItems:[]
    },
    
    // {
    //   title:'Insights',
    //   description:'Insights',
    //   iconClass:'fa fa-fw bg-insight',
    //   url:'/insights',
    //   roles:['bu','ds'],
    //   subItems:[]
    // },
    {
      title:'Model Evaluation',
      description:'Model Evaluation',
      iconClass:'fa fa-fw bg-insight',
      url:'/evaluation',
      roles:['ds'],
      subItems:[]
    },
    {
      title:'API Endpoints',
      description:'API Endpoints',
      iconClass:'fa fa-fw bg-fa fa-exchange',
      url:'/api-endpoints',
      roles:['ds'],
      subItems:[]
    },

    {
      title:'Stored Videos',
      description:'Stored Videos',
      iconClass:'fa fa-fw bg-insight',
      url:'\/processtracking',
      roles:['ds','bu'],      
      subItems:[]
    },

    {
      title:'Cycles and Tags',
      description:'Cycles and Tags',
      iconClass:'fa fa-fw bg-label',
      url:'\/processtracking',
      roles:['ds','bu'],      
      subItems:[]
    },

    {
      title:'Video Analytics',
      description:'Video Analytics',
      iconClass:'fa fa-fw bg-pridiction',
      url:'\/processtracking',
      roles:['ds','bu'],      
      subItems:[]
    },

  ]

const navTopBarItems=[
    {
        title:'Insights',
        description:'Insights',
        iconClass:'fa fa-fw bg-insight',
        url:'#',
        subItems:[]  
    }
]

const appAlertItems = [
    {
        title:'Status Update',
        url:'#',
        titleClass:'text-success',
        titleIcon:'fa fa-long-arrow-up fa-fw',
        alertTime:'11:21 AM',
        description: 'This is an automated server response message. All systems are online'      
    },
    {
        title:'Status Update',
        url:'#',
        titleClass:'text-danger',
        titleIcon:'fa fa-long-arrow-down fa-fw',
        alertTime:'11:27 AM',
        description: 'This is an automated server response message. All systems are have gone down'      
    },
]

const appMsgItems = [
    {
        name:'David Miller',
        url:'#',
        msgTime:'11:21 AM',
        data: 'Hey there! This new version of SB Admin is pretty awesome! These messages clip off when they reach the end of the box so they don\'t overflow over to the sides!'      
    },
    {
        name:'Jane Smith',
        url:'#',
        msgTime:'11:21 AM',
        data: 'I was wondering if you could meet for an appointment at 3:00 instead of 4:00. Thanks!'      
    },
    {
        name:'John Doe',
        url:'#',
        msgTime:'11:21 AM',
        data: 'I\'ve sent the final files over to you for review. When you\'re able to sign off of them let me know and we can discuss distribution.'      
    },
]

export {navSideBarItem,navTopBarItems,homePageAppItems,appAlertItems,appMsgItems}