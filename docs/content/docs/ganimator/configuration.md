---
title: Configuration
weight: 2
---

`...`

<!--more-->

## Heading

### Sub

```yaml {filename="test.yaml"}
# @package _global_

_moai_:
  _definitions_:
    _collections_:
      _metrics_:
        features:
          coverage:
            pred: [gen_feats]
            gt: [gt_feats]
            _out_: [coverage]
          gdiv:
            pred: [gen_feats]
            gt: [gt_feats]
            _out_: [ganimator_gdiv]
          ldiv:
            pred: [gen_feats]
            gt: [gt_feats]
            _out_: [ganimator_ldiv]
          mdm_gdiv:
            pred: [motion_embed]
            gt: [motion_embed_gt]
            _out_: [mdm_gdiv]
          mdm_ldiv:
            pred: [clips_embeds]
            gt: [clips_embeds_gt]
            _out_: [mdm_ldiv]
```

There are different types of `...`:

1. One
    ```yaml
    - name: name
      pageRef: /ref
    ```
2. Two

3. Three
    ```yaml
    - name: name
      params:
        type: type
    ```

{{< filetree/container >}}
  {{< filetree/folder name="conf/project" >}}
    {{< filetree/file name="data.yaml" >}}
    {{< filetree/file name="main.yaml" >}}
    {{< filetree/file name="flow.yaml" >}}
  {{< /filetree/folder >}}
{{< /filetree/container >}}