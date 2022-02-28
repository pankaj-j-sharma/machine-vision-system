import { ComponentFixture, TestBed } from '@angular/core/testing';

import { LabellingComponent } from './labelling.component';

describe('LabellingComponent', () => {
  let component: LabellingComponent;
  let fixture: ComponentFixture<LabellingComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ LabellingComponent ]
    })
    .compileComponents();
  });

  beforeEach(() => {
    fixture = TestBed.createComponent(LabellingComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
