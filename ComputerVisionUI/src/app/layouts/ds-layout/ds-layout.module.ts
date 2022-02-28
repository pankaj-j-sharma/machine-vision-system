import { NgModule } from "@angular/core";
import { RouterModule } from "@angular/router";
import { DSLayoutRoutes } from "./ds-layout.routing";


@NgModule({
  imports: [
    RouterModule.forChild(DSLayoutRoutes),
  ],
  exports:[RouterModule],
  declarations: [],
  providers:[
  ]
})
export class DSLayoutModule {}